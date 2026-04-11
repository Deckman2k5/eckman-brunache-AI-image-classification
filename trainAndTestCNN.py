import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import CSVLogger
import os

mixed_precision.set_global_policy('mixed_float16')

# --- 1. Configuration ---
# Can be changed to use more or less compute
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15

# Define data directories
TRAIN_DIR = 'Images/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/train'
TEST_DIR = 'Images/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/test'

# --- 2. Data Loading ---
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary' # binary since we have two classes: AI vs Real
)

print("Loading testing data...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Optimize dataset loading for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Model Architecture ---
def build_cnn_model():
    model = models.Sequential([
        # Rescaling layer to normalize pixel values from [0, 255] to [0, 1]
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Helps prevent overfitting
        
        # Output Layer: 1 neuron with sigmoid for binary classification (0 or 1)
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_cnn_model()
model.summary()

# --- 4. Compilation ---
# We include Accuracy, Precision, and Recall in our metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# --- 5. Callbacks ---
# ModelCheckpoint ensures progress is saved. 
# It saves the best model based on validation accuracy so you don't lose progress if interrupted.
checkpoint_path = "ai_image_detector_best.keras"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False, # Saves the entire model architecture + weights
    monitor='val_accuracy',  # Monitor performance on the test set
    mode='max',
    save_best_only=True,     # Only overwrite if the model improved
    verbose=1
)

csv_logger = CSVLogger('training_log.csv', append=False)

# --- 6. Training ---
print("\nStarting training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback, csv_logger]
)

# --- 7. Final Evaluation ---
print("\n--- Final Model Evaluation on Testing Set ---")
# Load the best weights saved by the checkpoint to ensure we evaluate the best epoch
best_model = tf.keras.models.load_model(checkpoint_path)

results = best_model.evaluate(test_dataset, verbose=0)

# results is a list: [loss, accuracy, precision, recall]
print(f"Test Loss:      {results[0]:.4f}")
print(f"Test Accuracy:  {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]*100:.2f}%")
print(f"Test Recall:    {results[3]*100:.2f}%")