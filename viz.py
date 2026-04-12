import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)
import os

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
NPZ_PATH         = "resnet50_inference_results.npz"
HISTORY_CSV_PATH = "resnet50_finetune_log.csv"
OUTPUT_DIR       = "eval_results"
THRESHOLD        = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Load inference results from NPZ
# ─────────────────────────────────────────────
print("Loading inference results from NPZ...")
data        = np.load(NPZ_PATH, allow_pickle=True)
all_probs   = data['probs']
all_labels  = data['labels'].astype(int)
all_images  = data['images']
class_names = data['class_names'].tolist()
all_preds   = (all_probs >= THRESHOLD).astype(int)

print(f"Loaded {len(all_labels)} samples")
print(f"Classes: {class_names}")

# ─────────────────────────────────────────────
# 3. Console summary
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("  EVALUATION SUMMARY")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=class_names))

for idx, name in enumerate(class_names):
    mask = all_labels == idx
    acc  = (all_preds[mask] == all_labels[mask]).mean()
    print(f"  Accuracy on '{name}': {acc*100:.2f}%")

# ─────────────────────────────────────────────
# 4. Confusion Matrix
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
# 5. ROC Curve
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
# 6. Precision–Recall Curve
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
# 7. Confidence distribution histogram
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
# 8. Sample prediction grid (correct vs wrong)
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

# ─────────────────────────────────────────────
# 9. Training History (from CSV)
# ─────────────────────────────────────────────
print("\nLoading training history...")
df = pd.read_csv(HISTORY_CSV_PATH)
epochs = df['epoch'] + 1

# 9a. Accuracy curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, df['accuracy'],     lw=2, label='Train Accuracy')
ax.plot(epochs, df['val_accuracy'], lw=2, label='Val Accuracy', linestyle='--')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# 9b. Loss curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, df['loss'],     lw=2, label='Train Loss')
ax.plot(epochs, df['val_loss'], lw=2, label='Val Loss', linestyle='--')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# 9c. Overfitting gap
gap = df['accuracy'] - df['val_accuracy']
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, gap, lw=2, color='crimson', label='Accuracy Gap (Train - Val)')
ax.axhline(0, color='gray', linestyle='--', lw=1)
ax.fill_between(epochs, gap, alpha=0.15, color='crimson')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy Gap")
ax.set_title("Overfitting Gap Over Training", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "overfitting_gap.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# 9d. Precision & Recall curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(epochs, df['precision'],     lw=2, label='Train Precision')
axes[0].plot(epochs, df['val_precision'], lw=2, label='Val Precision', linestyle='--')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision Over Epochs", fontsize=13, fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, df['recall'],     lw=2, label='Train Recall')
axes[1].plot(epochs, df['val_recall'], lw=2, label='Val Recall', linestyle='--')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Recall")
axes[1].set_title("Recall Over Epochs", fontsize=13, fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "precision_recall_over_epochs.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

# 9e. Combined 2x2 dashboard
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training History Dashboard", fontsize=16, fontweight="bold")

axes[0, 0].plot(epochs, df['accuracy'],     lw=2, label='Train')
axes[0, 0].plot(epochs, df['val_accuracy'], lw=2, label='Val', linestyle='--')
axes[0, 0].set_title("Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(epochs, df['loss'],     lw=2, label='Train')
axes[0, 1].plot(epochs, df['val_loss'], lw=2, label='Val', linestyle='--')
axes[0, 1].set_title("Loss")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(epochs, df['precision'],     lw=2, label='Train')
axes[1, 0].plot(epochs, df['val_precision'], lw=2, label='Val', linestyle='--')
axes[1, 0].set_title("Precision")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(epochs, df['recall'],     lw=2, label='Train')
axes[1, 1].plot(epochs, df['val_recall'], lw=2, label='Val', linestyle='--')
axes[1, 1].set_title("Recall")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "training_dashboard.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"Saved: {path}")

print(f"\nAll evaluation outputs saved to '{OUTPUT_DIR}/'")
print("="*50)