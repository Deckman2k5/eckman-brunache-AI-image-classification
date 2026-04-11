import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)
import os

# ─────────────────────────────────────────────
# 1. Configuration  (match your training script)
# ─────────────────────────────────────────────
IMG_HEIGHT  = 128
IMG_WIDTH   = 128
BATCH_SIZE  = 32
TEST_DIR    = "SmallImages/test"
MODEL_PATH  = "ai_image_detector_best.h5"
OUTPUT_DIR  = "eval_results"          # all plots saved here
THRESHOLD   = 0.5                     # decision boundary (tweak if needed)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Load model & test data
# ─────────────────────────────────────────────
print("Loading model...")
with custom_object_scope({'Policy': mixed_precision.Policy}):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("Loading test dataset...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False           # keep order so images align with predictions
)

class_names = test_dataset.class_names   # e.g. ['AI', 'Real']
print(f"Classes detected: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
test_dataset_perf = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ─────────────────────────────────────────────
# 3. Collect predictions & ground-truth labels
# ─────────────────────────────────────────────
print("Running inference...")
all_probs  = []   # raw sigmoid outputs  [0.0 – 1.0]
all_labels = []   # ground-truth integers
all_images = []   # pixel arrays for the sample grid

for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    all_probs.append(preds.flatten())
    all_labels.append(labels.numpy().flatten().astype(int))
    all_images.append(images.numpy())

all_probs  = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
all_images = np.concatenate(all_images)
all_preds  = (all_probs >= THRESHOLD).astype(int)

# ─────────────────────────────────────────────
# 4. Console summary
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("  EVALUATION SUMMARY")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=class_names))

# Per-class accuracy
for idx, name in enumerate(class_names):
    mask = all_labels == idx
    acc  = (all_preds[mask] == all_labels[mask]).mean()
    print(f"  Accuracy on '{name}': {acc*100:.2f}%")

# ─────────────────────────────────────────────
# 5. Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"\nSaved: {path}")

# ─────────────────────────────────────────────
# 6. ROC Curve
# ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ─────────────────────────────────────────────
# 7. Precision–Recall Curve
# ─────────────────────────────────────────────
precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)
ap = average_precision_score(all_labels, all_probs)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall_vals, precision_vals, lw=2, label=f"AP = {ap:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ─────────────────────────────────────────────
# 8. Confidence distribution histogram
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for idx, name in enumerate(class_names):
    mask = all_labels == idx
    ax.hist(all_probs[mask], bins=40, alpha=0.6, label=name)
ax.axvline(THRESHOLD, color="red", linestyle="--", label=f"Threshold = {THRESHOLD}")
ax.set_xlabel("Predicted Probability (class index 1)")
ax.set_ylabel("Count")
ax.set_title("Confidence Score Distribution", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "confidence_distribution.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# ─────────────────────────────────────────────
# 9. Sample prediction grid  (correct vs wrong)
# ─────────────────────────────────────────────
def sample_grid(title, indices, n=16):
    """Plot up to n images from the given indices."""
    indices = indices[:n]
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = all_images[idx].astype(np.uint8)
        ax.imshow(img)
        true_name = class_names[all_labels[idx]]
        pred_name = class_names[all_preds[idx]]
        conf      = all_probs[idx]
        ax.set_title(f"T:{true_name}\nP:{pred_name} ({conf:.2f})", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    return fig

correct_idx   = np.where(all_preds == all_labels)[0]
incorrect_idx = np.where(all_preds != all_labels)[0]

# Shuffle for variety
rng = np.random.default_rng(42)
rng.shuffle(correct_idx)
rng.shuffle(incorrect_idx)

fig = sample_grid("Correct Predictions (sample)", correct_idx)
path = os.path.join(OUTPUT_DIR, "correct_predictions.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

if len(incorrect_idx) > 0:
    fig = sample_grid("Misclassified Images (sample)", incorrect_idx)
    path = os.path.join(OUTPUT_DIR, "misclassified.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
else:
    print("No misclassified images found — perfect score!")

print(f"\nAll evaluation outputs saved to '{OUTPUT_DIR}/'")
print("="*50)