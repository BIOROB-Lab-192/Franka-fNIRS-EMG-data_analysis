from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

FIG_DIR = Path("./machine_learning/fusion_pertask/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

EMG_PATH = "./machine_learning/EMG_pertask/export/predictions_task_level_loocv.csv"
FNIRS_PATH = "./machine_learning/fNIRS_pertask/export/predictions_task_level_loocv.csv"

W_FNIRS = 0.3
W_EMG = 0.7

emg = pd.read_csv(EMG_PATH)
fnirs = pd.read_csv(FNIRS_PATH)

prob_cols = [c for c in emg.columns if c.startswith("prob_")]

emg_small = emg[["held_out_run", "task_instance", "true_label", "true_task"] + prob_cols].copy()
fnirs_small = fnirs[["held_out_run", "task_instance", "true_label", "true_task"] + prob_cols].copy()

df = emg_small.merge(
    fnirs_small,
    on=["held_out_run", "task_instance"],
    suffixes=("_emg", "_fnirs"),
)

if not (df["true_label_emg"] == df["true_label_fnirs"]).all():
    raise ValueError("EMG and fNIRS true labels do not match.")

df["true_label"] = df["true_label_emg"]
df["true_task"] = df["true_task_emg"]

task_names = [c.replace("prob_", "") for c in prob_cols]

for task in task_names:
    df[f"prob_{task}_fused"] = (
        W_EMG * df[f"prob_{task}_emg"]
        + W_FNIRS * df[f"prob_{task}_fnirs"]
    )

fused_prob_cols = [f"prob_{task}_fused" for task in task_names]

df["pred_label_fused"] = df[fused_prob_cols].values.argmax(axis=1)
df["pred_task_fused"] = [task_names[i] for i in df["pred_label_fused"]]
df["correct_fused"] = df["pred_label_fused"] == df["true_label"]

print("Accuracy:", accuracy_score(df["true_label"], df["pred_label_fused"]))
print("Balanced accuracy:", balanced_accuracy_score(df["true_label"], df["pred_label_fused"]))
print("Macro F1:", f1_score(df["true_label"], df["pred_label_fused"], average="macro"))
print()
print(classification_report(df["true_label"], df["pred_label_fused"], target_names=task_names))

# ───────────────────── CONFUSION MATRIX ─────────────────────

# task_names comes from prob_cols order, which matches argmax ordering
task_names_sorted = task_names  # already in the order the argmax was computed
n_classes = len(task_names_sorted)

y_true = df["true_label"].values
y_pred = df["pred_label_fused"].values

cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

fig_size = max(7, n_classes * 0.7)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(n_classes),
    yticks=np.arange(n_classes),
    xticklabels=task_names_sorted,
    yticklabels=task_names_sorted,
    ylabel="True task",
    xlabel="Predicted task",
    title="Fused (EMG + fNIRS) Confusion Matrix — LOOCV",
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

thresh = cm.max() / 2 if cm.max() > 0 else 0
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9,
        )

plt.tight_layout()
fig.savefig(FIG_DIR / "confusion_matrix_fused_pertask.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved confusion matrix: {FIG_DIR / 'confusion_matrix_fused_pertask.png'}")

# ───────────────────── PER-CATEGORY ACCURACY ─────────────────────

per_cat_acc = []
for cat_idx in range(n_classes):
    mask = y_true == cat_idx
    if mask.sum() > 0:
        per_cat_acc.append((task_names_sorted[cat_idx], y_pred[mask].tolist().count(cat_idx) / mask.sum()))
    else:
        per_cat_acc.append((task_names_sorted[cat_idx], 0.0))

cat_names = [c[0] for c in per_cat_acc]
cat_accs = [c[1] for c in per_cat_acc]
mean_acc = np.mean(cat_accs)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(cat_names)), cat_accs)
ax.axhline(y=mean_acc, linestyle="--", label=f"Mean: {mean_acc:.3f}")
ax.set_xlabel("Task")
ax.set_ylabel("Accuracy")
ax.set_title("Per-Category Accuracy — Fused (EMG + fNIRS)")
ax.set_xticks(range(len(cat_names)))
ax.set_xticklabels(cat_names, rotation=45, ha="right", fontsize=9)
ax.set_ylim([0, 1.05])
ax.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "per_category_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved per-category accuracy: {FIG_DIR / 'per_category_accuracy.png'}")

df.to_csv("./machine_learning/fusion_pertask/task_late_fusion_simple.csv", index=False)