import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import mixed_precision
import numpy as np
import os

# ─────────────────────────────────────────────
# 0. Reproducibility
# ─────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. GPU Check
# ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    # mixed_float16 uses float16 for computation and float32 for weights,
    # significantly speeding up training on modern GPUs with minimal accuracy loss
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled (float16)")
else:
    print("WARNING: No GPU detected — running on CPU. Training will be slow.")

# Confirm which device TensorFlow will use for training
print(f"Training device: {'GPU' if gpus else 'CPU'}")

# ─────────────────────────────────────────────
# 2. Configuration
# ─────────────────────────────────────────────
IMG_HEIGHT      = 224     # ResNet50 was designed for 224x224 inputs
IMG_WIDTH       = 224
BATCH_SIZE      = 32      # 32 is a standard default; larger batches train faster
                          # but require more VRAM and can reduce generalization
EPOCHS_PHASE1   = 15      # Phase 1: train only the classifier head
EPOCHS_PHASE2   = 10      # Phase 2: fine-tune top ResNet layers (fewer needed
                          # since weights are already close to optimal)
VAL_SPLIT       = 0.2     # 20% of training data reserved for validation
FINE_TUNE_LAYERS = 30     # Number of top ResNet50 layers to unfreeze in phase 2.
                          # Earlier layers detect basic edges/textures that are
                          # universal — only later layers need task-specific tuning

TRAIN_DIR = 'SmallImages/train'
TEST_DIR  = 'SmallImages/test'

# ─────────────────────────────────────────────
# 3. Data Loading
# ─────────────────────────────────────────────

# Check class balance before training — imbalanced classes lead to misleading
# accuracy scores where the model just predicts the majority class
print("\nChecking class balance...")
for split_dir in [TRAIN_DIR, TEST_DIR]:
    split_name = os.path.basename(split_dir)
    for class_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"  {split_name}/{class_name}: {count} images")

print("\nLoading training data...")
# seed=42 ensures the validation split is consistent across runs
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split=VAL_SPLIT,   # reserve 20% for validation
    subset='training',
    seed=42
)

print("Loading validation data...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split=VAL_SPLIT,
    subset='validation',
    seed=42
)

print("Loading test data...")
# Test set is completely separate — never seen during training or validation
# This gives an unbiased measure of real-world performance
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False   # must be False so predictions align with labels in eval.py
)

class_names = train_dataset.class_names
print(f"\nClasses: {class_names}")

# ─────────────────────────────────────────────
# 4. Data Augmentation
# ─────────────────────────────────────────────
# Augmentation artificially expands the training set by applying random
# transformations. This forces the model to learn general features rather
# than memorizing specific training images, reducing overfitting.
# Applied only to training data — validation and test are never augmented.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),      # AI images can appear mirrored
    layers.RandomRotation(0.1),           # slight rotation invariance
    layers.RandomZoom(0.1),               # slight scale invariance
    layers.RandomBrightness(0.1),         # lighting variation robustness
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = (
    train_dataset
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000, seed=42)
    .prefetch(buffer_size=AUTOTUNE)
)
val_dataset  = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ─────────────────────────────────────────────
# 5. Model Architecture
# ─────────────────────────────────────────────
def build_resnet_model():
    # Use the Functional API instead of Sequential for cleaner architecture
    # and easier access to named layers
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_image")

    # ResNet50's preprocess_input scales pixels from [0,255] to the range
    # the model was trained on during ImageNet pretraining
    x = tf.keras.applications.resnet50.preprocess_input(inputs)

    # Load ResNet50 pretrained on ImageNet
    # include_top=False removes the final 1000-class classification layer
    # so we can attach our own binary classifier
    base_model = tf.keras.applications.ResNet50(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze all layers for phase 1

    x = base_model(x, training=False)  # training=False keeps BatchNorm frozen
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation='relu', name="dense_head")(x)
    x = layers.Dropout(0.5, name="dropout")(x)

    # dtype='float32' ensures the output is float32 even with mixed precision
    # enabled — sigmoid outputs need full precision for stable binary crossentropy
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name="output")(x)

    return tf.keras.Model(inputs, outputs), base_model

model, base_model = build_resnet_model()
model.summary()

# ─────────────────────────────────────────────
# 6. Phase 1 — Train classifier head only
# ─────────────────────────────────────────────
# learning_rate=1e-3 is the Adam default and works well when the base is
# frozen since only the small classifier head is being updated
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

checkpoint_path = "resnet50_best.keras"

phase1_callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    # patience=3 means stop if val_accuracy doesn't improve for 3 consecutive
    # epochs — prevents wasting compute and reduces overfitting risk
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    CSVLogger('resnet50_training_log.csv', append=False)
]

print("\nPhase 1: Training classifier head (ResNet50 base frozen)...")
history_phase1 = model.fit(
    train_dataset,
    validation_data=val_dataset,   # validation set — separate from test
    epochs=EPOCHS_PHASE1,
    callbacks=phase1_callbacks
)

# ─────────────────────────────────────────────
# 7. Phase 2 — Fine-tune top ResNet layers
# ─────────────────────────────────────────────
print(f"\nPhase 2: Unfreezing top {FINE_TUNE_LAYERS} layers for fine-tuning...")

base_model.trainable = True

# Freeze all layers except the last FINE_TUNE_LAYERS
# Early layers (edges, textures, colors) are universal and don't need updating.
# Later layers (complex shapes, object parts) are more task-specific
# and benefit from fine-tuning on our AI vs Real dataset.
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable ResNet layers: {trainable_count} / {len(base_model.layers)}")

# learning_rate=1e-5 is 100x smaller than phase 1. This is critical —
# a large learning rate would destroy the pretrained weights that took
# ImageNet training to build up, causing catastrophic forgetting.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

phase2_callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    # append=False starts a fresh CSV for phase 2 to avoid duplicate epoch numbers
    # that would break the training history graphs in eval.py
    CSVLogger('resnet50_finetune_log.csv', append=False)
]

print("Phase 2: Fine-tuning top ResNet50 layers...")
history_phase2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE2,
    callbacks=phase2_callbacks
)

# ─────────────────────────────────────────────
# 8. Final Evaluation on held-out test set
# ─────────────────────────────────────────────
print("\n--- Final Evaluation on Test Set ---")
print("(Test set was never used during training or validation)")

best_model = tf.keras.models.load_model(checkpoint_path)
results = best_model.evaluate(test_dataset, verbose=0)

print(f"Test Loss:      {results[0]:.4f}")
print(f"Test Accuracy:  {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]*100:.2f}%")
print(f"Test Recall:    {results[3]*100:.2f}%")

# ─────────────────────────────────────────────
# 9. Export inference results for eval.py
# ─────────────────────────────────────────────
print("\nExporting inference results...")
raw_test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

all_probs  = []
all_labels = []
all_images = []

for images, labels in raw_test_dataset:
    preds = best_model.predict(images, verbose=0)
    all_probs.append(preds.flatten())
    all_labels.append(labels.numpy().flatten().astype(int))
    all_images.append(images.numpy())

all_probs  = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
all_images = np.concatenate(all_images)

np.savez_compressed(
    "resnet50_inference_results.npz",
    probs=all_probs,
    labels=all_labels,
    images=all_images,
    class_names=np.array(class_names)
)
print("Saved resnet50_inference_results.npz")
print("\nDone! Files for eval.py:")
print("  - resnet50_inference_results.npz")
print("  - resnet50_training_log.csv   (phase 1)")
print("  - resnet50_finetune_log.csv   (phase 2)")