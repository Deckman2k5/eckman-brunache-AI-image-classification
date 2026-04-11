import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import mixed_precision
import numpy as np
import json
import os

# ─────────────────────────────────────────────
# 0. GPU Check
# ─────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus}")
    mixed_precision.set_global_policy('mixed_float16')
else:
    print("No GPU detected — running on CPU. Training will be slower.")

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
IMG_HEIGHT = 224      # ResNet50 expects 224x224
IMG_WIDTH  = 224
BATCH_SIZE = 32
EPOCHS     = 15

TRAIN_DIR  = 'SmallImages/train'
TEST_DIR   = 'SmallImages/test'

# ─────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    seed=42
)

print("Loading test data...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

class_names = train_dataset.class_names
print(f"Classes: {class_names}")

# Data augmentation — applied only during training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = (
    train_dataset
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=AUTOTUNE)
)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ─────────────────────────────────────────────
# 3. Model Architecture (ResNet50 + custom head)
# ─────────────────────────────────────────────
def build_resnet_model():
    # Load ResNet50 pretrained on ImageNet, without the top classification layer
    base_model = tf.keras.applications.ResNet50(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,       # removes ImageNet classification head
        weights='imagenet'       # loads pretrained weights
    )
    base_model.trainable = False # freeze all pretrained layers

    model = models.Sequential([
        # ResNet50 expects inputs preprocessed to its specific range
        layers.Lambda(tf.keras.applications.resnet50.preprocess_input),

        base_model,

        # Replace the original head with our binary classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')  # float32 for stability
    ])
    return model

model = build_resnet_model()
model.summary()

# ─────────────────────────────────────────────
# 4. Compilation
# ─────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# ─────────────────────────────────────────────
# 5. Callbacks
# ─────────────────────────────────────────────
checkpoint_path = "resnet50_best.keras"

callbacks = [
    # Save best model based on val_accuracy
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    # Stop early if val_accuracy stops improving for 3 epochs
    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    # Save training history to CSV
    CSVLogger('resnet50_training_log.csv', append=False)
]

# ─────────────────────────────────────────────
# 6. Phase 1 — Train with frozen base
# ─────────────────────────────────────────────
print("\nPhase 1: Training classifier head (base frozen)...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ─────────────────────────────────────────────
# 7. Phase 2 — Fine-tune (unfreeze top layers)
# ─────────────────────────────────────────────
print("\nPhase 2: Fine-tuning top layers of ResNet50...")

base_model = model.layers[1]  # ResNet50 is the second layer (after Lambda)
base_model.trainable = True

# Only unfreeze the last 30 layers — earlier layers detect basic features
# that are already well trained and don't need updating
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Lower learning rate for fine-tuning to avoid destroying pretrained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

fine_tune_callbacks = [
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
    CSVLogger('resnet50_training_log.csv', append=True)  # append to same CSV
]

history_fine = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,  # fine-tuning needs fewer epochs
    callbacks=fine_tune_callbacks
)

# ─────────────────────────────────────────────
# 8. Final Evaluation
# ─────────────────────────────────────────────
print("\n--- Final Evaluation ---")
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
print("\nDone! Use resnet50_training_log.csv and resnet50_inference_results.npz with eval.py")