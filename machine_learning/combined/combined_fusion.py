"""
fNIRS + EMG Late Fusion — Proof of Concept
============================================
Loads both pretrained models, runs inference on all data,
tests fusion strategies. Data leakage: known and accepted (PoC).
"""

import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path

# ──────────────────────────── CONFIG ────────────────────────────

FNIRS_DATA = Path("./data/processed/combined/data_packet/fnirs_full.parquet")
EMG_DATA = Path("./data/processed/combined/data_packet/emg_rms.parquet")
FNIRS_MODEL_PATH = Path("./machine_learning/fNIRS_perrun/export/model.pt")
EMG_MODEL_PATH = Path("./machine_learning/EMG_perrun/export/prod_model.pt")
EXPORT_DIR = Path("./machine_learning/combined/export")
FIG_DIR = Path("./machine_learning/combined/figures")

DEVICE = "cpu"
BATCH_SIZE = 64  # inference only — can use large batch

# fNIRS: 52 channels (26 HBO + 26 HBR)
FNIRS_COLUMNS = [
    "S1_D1_hbo", "S1_D2_hbo", "S1_D3_hbo", "S1_D8_hbo",
    "S2_D1_hbo", "S2_D3_hbo", "S2_D4_hbo", "S2_D9_hbo",
    "S3_D2_hbo", "S3_D3_hbo", "S3_D10_hbo",
    "S4_D3_hbo", "S4_D4_hbo", "S4_D11_hbo",
    "S5_D5_hbo", "S5_D6_hbo", "S5_D7_hbo", "S5_D12_hbo",
    "S6_D5_hbo", "S6_D7_hbo", "S6_D13_hbo",
    "S7_D6_hbo", "S7_D7_hbo", "S7_D14_hbo",
    "S8_D7_hbo", "S8_D15_hbo",
    "S1_D1_hbr", "S1_D2_hbr", "S1_D3_hbr", "S1_D8_hbr",
    "S2_D1_hbr", "S2_D3_hbr", "S2_D4_hbr", "S2_D9_hbr",
    "S3_D2_hbr", "S3_D3_hbr", "S3_D10_hbr",
    "S4_D3_hbr", "S4_D4_hbr", "S4_D11_hbr",
    "S5_D5_hbr", "S5_D6_hbr", "S5_D7_hbr", "S5_D12_hbr",
    "S6_D5_hbr", "S6_D7_hbr", "S6_D13_hbr",
    "S7_D6_hbr", "S7_D7_hbr", "S7_D14_hbr",
    "S8_D7_hbr", "S8_D15_hbr",
]

# EMG: 4 channels
EMG_COLUMNS = [
    "Avanti Sensor 1 (82703) | EMG 1 (mV)",
    "Avanti Sensor 2 (82529) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 2 (mV)",
]
EMG_DOWNSAMPLE = 5
EMG_TARGET_LEN = 5036

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

print("fNIRS + EMG Late Fusion — Proof of Concept")
print("=" * 60)


# ──────────────────────────── DATA LOADING ────────────────────────────

def load_runs(data_path, label_col="is_robot"):
    """Load parquet, build per-run dicts sorted by (run_id, task_instance)."""
    df = pl.read_parquet(data_path)
    runs = {}
    for run_id in df["run_id"].unique().sort().to_list():
        run_df = df.filter(pl.col("run_id") == run_id).sort("task_instance")
        label = int(run_df[label_col][0])
        n_inst = run_df["task_instance"].n_unique()
        runs[run_id] = {"df": run_df, "label": label, "n_instances": n_inst}
        print(f"  {run_id}: label={label}, {n_inst} instances")
    return runs


print("\nLoading fNIRS...")
fnirs_runs = load_runs(FNIRS_DATA)

print("\nLoading EMG...")
emg_runs = load_runs(EMG_DATA)

# Verify alignment
shared_runs = sorted(set(fnirs_runs.keys()) & set(emg_runs.keys()))
print(f"\nShared runs: {len(shared_runs)}")
assert len(shared_runs) == 10, f"Expected 10 shared runs, got {len(shared_runs)}"


# ──────────────────────────── PREPROCESSING ────────────────────────────

def preprocess_fnirs_window(window_df):
    """fNIRS: select 52 channels, sort by time, return (C, T)."""
    window_df = window_df.sort("time_index")
    raw = window_df.select(FNIRS_COLUMNS).to_numpy()
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw.T.astype(np.float32)  # (52, 201)


def preprocess_emg_window(window_df):
    """EMG: select 4 channels, downsample, pad/truncate to target, return (C, T)."""
    raw = window_df.select(EMG_COLUMNS).to_numpy()
    ds = raw[::EMG_DOWNSAMPLE]
    if ds.shape[0] >= EMG_TARGET_LEN:
        ds = ds[:EMG_TARGET_LEN]
    else:
        pad = EMG_TARGET_LEN - ds.shape[0]
        ds = np.pad(ds, ((0, pad), (0, 0)), mode="constant")
    ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
    return ds.T.astype(np.float32)  # (4, 5036)


def build_all_windows(runs, preprocess_fn):
    """Process all windows from all runs. Returns (windows, labels, run_ids, instances)."""
    all_windows, all_labels, all_run_ids, all_instances = [], [], [], []
    for run_id in sorted(runs.keys()):
        info = runs[run_id]
        run_df = info["df"]
        label = info["label"]
        instances = run_df["task_instance"].unique().sort().to_list()
        for inst in instances:
            inst_df = run_df.filter(pl.col("task_instance") == inst)
            all_windows.append(preprocess_fn(inst_df))
            all_labels.append(label)
            all_run_ids.append(run_id)
            all_instances.append(inst)
    return all_windows, all_labels, all_run_ids, all_instances


def normalize_windows(windows):
    """Global z-score normalization across all windows. Leaky (PoC)."""
    arr = np.array(windows)  # (N, C, T)
    mean = arr.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    std = arr.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32), mean, std


print("\nPreprocessing fNIRS windows...")
fnirs_windows, labels, run_ids, instances = build_all_windows(fnirs_runs, preprocess_fnirs_window)
fnirs_arr, fnirs_mean, fnirs_std = normalize_windows(fnirs_windows)
print(f"  {fnirs_arr.shape} — {fnirs_arr.shape[0]} windows, {fnirs_arr.shape[1]} channels, {fnirs_arr.shape[2]} timesteps")

print("\nPreprocessing EMG windows...")
emg_windows, _, _, _ = build_all_windows(emg_runs, preprocess_emg_window)
emg_arr, emg_mean, emg_std = normalize_windows(emg_windows)
print(f"  {emg_arr.shape} — {emg_arr.shape[0]} windows, {emg_arr.shape[1]} channels, {emg_arr.shape[2]} timesteps")

labels = np.array(labels)
run_ids = np.array(run_ids)
instances = np.array(instances)


# ──────────────────────────── DATASET ────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# ──────────────────────────── MODEL DEFINITIONS ────────────────────────────

class FNIRSClassifier1D(nn.Module):
    """1D CNN for fNIRS: input (batch, 52, 201)."""
    def __init__(self, n_channels=52, n_classes=2,
                 classifier_drop=0.3, conv_drop=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(classifier_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EMGClassifier1D(nn.Module):
    """1D CNN for EMG: input (batch, 4, 5036)."""
    def __init__(self, n_channels=4, n_classes=2,
                 classifier_drop=0.3, conv_drop=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(classifier_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────── WEIGHT LOADING ────────────────────────────

def load_model(model_class, weight_path, **kwargs):
    """Load model from saved weights."""
    model = model_class(**kwargs)
    state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


print("\nLoading pretrained models...")
fnirs_model = load_model(FNIRSClassifier1D, FNIRS_MODEL_PATH,
                         n_channels=52, classifier_drop=0.3, conv_drop=0.0)
emg_model = load_model(EMGClassifier1D, EMG_MODEL_PATH,
                       n_channels=4, classifier_drop=0.3, conv_drop=0.2)
print(f"  fNIRS: {sum(p.numel() for p in fnirs_model.parameters()):,} params")
print(f"  EMG:   {sum(p.numel() for p in emg_model.parameters()):,} params")


# ──────────────────────────── INFERENCE ────────────────────────────

def run_inference(model, windows, labels):
    """Run model on all windows, return prob_robot array."""
    ds = WindowDataset(windows, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob_robot
    return np.array(all_probs)


print("\nRunning inference...")
fnirs_probs = run_inference(fnirs_model, fnirs_arr, labels)
emg_probs = run_inference(emg_model, emg_arr, labels)
print(f"  fNIRS probs: {fnirs_probs.shape}, range [{fnirs_probs.min():.3f}, {fnirs_probs.max():.3f}]")
print(f"  EMG probs:   {emg_probs.shape}, range [{emg_probs.min():.3f}, {emg_probs.max():.3f}]")


# ──────────────────────────── FUSION STRATEGIES ────────────────────────────

def evaluate(name, probs, labels):
    """Compute metrics for a set of probability predictions."""
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1_r = f1_score(labels, preds, pos_label=1, zero_division=0)
    f1_n = f1_score(labels, preds, pos_label=0, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    return {"name": name, "acc": acc, "f1_r": f1_r, "f1_n": f1_n, "auc": auc}


# Individual baselines
print("\n" + "=" * 60)
print("EVALUATING FUSION STRATEGIES")
print("=" * 60)

results = []
results.append(evaluate("fNIRS only", fnirs_probs, labels))
results.append(evaluate("EMG only", emg_probs, labels))

# Strategy 1: Simple average
avg_probs = (fnirs_probs + emg_probs) / 2
results.append(evaluate("Simple Average", avg_probs, labels))

# Strategy 2: Weighted average (grid search)
best_w, best_acc = 0.5, 0
for w in np.arange(0, 1.05, 0.05):
    fused = w * fnirs_probs + (1 - w) * emg_probs
    acc = accuracy_score(labels, (fused > 0.5).astype(int))
    if acc > best_acc:
        best_acc, best_w = acc, w
weighted_probs = best_w * fnirs_probs + (1 - best_w) * emg_probs
r = evaluate(f"Weighted Avg (w={best_w:.2f})", weighted_probs, labels)
results.append(r)
print(f"  Best weight: w={best_w:.2f} (fNIRS) / {1-best_w:.2f} (EMG)")

# Strategy 3: Logistic regression
X_fuse = np.column_stack([fnirs_probs, emg_probs])
lr = LogisticRegression().fit(X_fuse, labels)
logistic_probs = lr.predict_proba(X_fuse)[:, 1]
results.append(evaluate("Logistic Regression", logistic_probs, labels))
print(f"  LR coefficients: fNIRS={lr.coef_[0][0]:.3f}, EMG={lr.coef_[0][1]:.3f}")

# Strategy 4: Max confidence
fnirs_pred = (fnirs_probs > 0.5).astype(int)
emg_pred = (emg_probs > 0.5).astype(int)
fnirs_conf = np.abs(fnirs_probs - 0.5)
emg_conf = np.abs(emg_probs - 0.5)
maxconf_pred = np.where(fnirs_conf >= emg_conf, fnirs_pred, emg_pred).astype(float)
results.append(evaluate("Max Confidence", maxconf_pred, labels))

# Print comparison table
print(f"\n{'Strategy':<25} {'Acc':>6} {'F1_R':>6} {'F1_N':>6} {'AUC':>6}")
print("-" * 55)
for r in sorted(results, key=lambda x: x["acc"], reverse=True):
    marker = " ★" if r["acc"] == max(x["acc"] for x in results) else ""
    print(f"{r['name']:<25} {r['acc']:>6.3f} {r['f1_r']:>6.3f} {r['f1_n']:>6.3f} {r['auc']:>6.3f}{marker}")

best = max(results, key=lambda x: x["acc"])
print(f"\nBest: {best['name']} → acc={best['acc']:.3f}")


# ──────────────────────────── PLOTS ────────────────────────────

print(f"\n{'='*60}")
print("SAVING PLOTS")
print(f"{'='*60}")

# Plot 1: Fusion comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
names = [r["name"] for r in results]
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

for ax, metric, title in zip(axes, ["acc", "f1_r", "auc"], ["Accuracy", "F1 (Robot)", "AUC"]):
    vals = [r[metric] for r in results]
    bars = ax.barh(range(len(names)), vals, color=colors[:len(names)], edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    ax.axvline(x=max(vals), color="gray", linestyle="--", alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v:.3f}",
                va="center", fontsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "fusion_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fusion_comparison.png")

# Plot 2: ROC overlay
fig, ax = plt.subplots(figsize=(7, 6))
roc_styles = [
    ("fNIRS only", fnirs_probs, "#3498db", "-"),
    ("EMG only", emg_probs, "#e74c3c", "-"),
    ("Simple Average", avg_probs, "#2ecc71", "--"),
    (f"Weighted Avg", weighted_probs, "#f39c12", "--"),
    ("Logistic Regression", logistic_probs, "#9b59b6", "--"),
]
for name, probs, color, ls in roc_styles:
    valid = ~np.isnan(probs)
    if valid.sum() > 1 and len(np.unique(labels[valid])) > 1:
        fpr, tpr, _ = roc_curve(labels[valid], probs[valid])
        auc_val = roc_auc_score(labels[valid], probs[valid])
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{name} (AUC={auc_val:.3f})")

ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":", label="Chance")
ax.set_xlim([0, 1.0]); ax.set_ylim([0, 1.05])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Fusion Comparison")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(FIG_DIR / "roc_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: roc_overlay.png")

# Plot 3: Confusion matrices — fNIRS vs EMG vs best fusion
best_name = best["name"]
if best_name == "Simple Average":
    best_probs = avg_probs
elif best_name.startswith("Weighted"):
    best_probs = weighted_probs
elif best_name == "Logistic Regression":
    best_probs = logistic_probs
elif best_name == "Max Confidence":
    best_probs = maxconf_pred
else:
    best_probs = fnirs_probs if "fNIRS" in best_name else emg_probs

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, probs, title in zip(axes,
    [fnirs_probs, emg_probs, best_probs],
    ["fNIRS only", "EMG only", f"Best: {best_name}"]):
    preds = (probs > 0.5).astype(int) if probs.max() <= 1.0 else probs.astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["No-Robot", "Robot"], yticklabels=["No-Robot", "Robot"],
           ylabel="True", xlabel="Predicted", title=title)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold")

fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
plt.tight_layout()
fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: confusion_matrices.png")


# ──────────────────────────── EXPORT ────────────────────────────

print(f"\n{'='*60}")
print("SAVING EXPORTS")
print(f"{'='*60}")

# Predictions CSV
pred_path = EXPORT_DIR / "predictions.csv"
pred_fields = [
    "run_id", "task_instance", "true_label",
    "prob_robot_fnirs", "prob_robot_emg",
    "pred_avg", "prob_avg",
    "pred_weighted", "prob_weighted",
    "pred_logistic", "prob_logistic",
    "pred_maxconf",
]
with open(pred_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=pred_fields)
    writer.writeheader()
    for i in range(len(labels)):
        writer.writerow({
            "run_id": run_ids[i],
            "task_instance": int(instances[i]),
            "true_label": int(labels[i]),
            "prob_robot_fnirs": f"{fnirs_probs[i]:.6f}",
            "prob_robot_emg": f"{emg_probs[i]:.6f}",
            "pred_avg": int(avg_probs[i] > 0.5),
            "prob_avg": f"{avg_probs[i]:.6f}",
            "pred_weighted": int(weighted_probs[i] > 0.5),
            "prob_weighted": f"{weighted_probs[i]:.6f}",
            "pred_logistic": int(logistic_probs[i] > 0.5),
            "prob_logistic": f"{logistic_probs[i]:.6f}",
            "pred_maxconf": int(maxconf_pred[i]),
        })
print(f"  Saved: {pred_path} ({len(labels)} rows)")

# Metrics CSV
csv_path = EXPORT_DIR / "metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["strategy", "accuracy", "f1_robot", "f1_norobot", "auc"])
    for r in sorted(results, key=lambda x: x["acc"], reverse=True):
        writer.writerow([r["name"], f"{r['acc']:.4f}", f"{r['f1_r']:.4f}",
                         f"{r['f1_n']:.4f}", f"{r['auc']:.4f}"])
print(f"  Saved: {csv_path}")

# Config JSON
config = {
    "fnirs_model": str(FNIRS_MODEL_PATH),
    "emg_model": str(EMG_MODEL_PATH),
    "fnirs_data": str(FNIRS_DATA),
    "emg_data": str(EMG_DATA),
    "n_windows": len(labels),
    "n_runs": len(shared_runs),
    "shared_runs": shared_runs,
    "fusion_strategies": ["average", "weighted_avg", "logistic", "max_confidence"],
    "best_strategy": best["name"],
    "best_accuracy": round(best["acc"], 4),
    "weighted_avg_w_fnirs": round(best_w, 2),
    "logistic_coef": {"fnirs": round(float(lr.coef_[0][0]), 4),
                      "emg": round(float(lr.coef_[0][1]), 4)},
    "note": "PoC with data leakage — trained and evaluated on same data",
}
with open(EXPORT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved: {EXPORT_DIR / 'config.json'}")

print(f"\n{'='*60}")
print("DONE ✓")
print(f"{'='*60}")
