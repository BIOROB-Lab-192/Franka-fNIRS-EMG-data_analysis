"""
EMG 1D CNN — LOOCV Evaluation + Final Production Model
======================================================

Dataset structure:
  - multiple runs
  - each run contains multiple task windows / task_instance values
  - each run has a run-level label: is_robot

This script does two separate things:

1. LOOCV evaluation by run_id
   - Hold out one full run at a time.
   - Train only on the other runs.
   - Normalize using training runs only.
   - Do NOT use the held-out run for early stopping or LR scheduling.
   - Report window-level and run-level out-of-fold metrics.

2. Final production training
   - After LOOCV is complete, train one final model on all runs.
   - Compute final normalization stats from all runs.
   - Save final model weights, normalization stats, downsampling metadata, and config.

Important:
  The LOOCV fold models are temporary evaluation models.
  The saved production model is trained on all available runs.
"""

import csv
import json
from pathlib import Path
from datetime import datetime

import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)


# ──────────────────────────── CONFIG ────────────────────────────

DATA_PATH = Path("./data/processed/combined/data_packet/emg_rms.parquet")
FIG_DIR = Path("./machine_learning/EMG_perrun/figures")
EXPORT_DIR = Path("./machine_learning/EMG_perrun/export")

EMG_COLUMNS = [
    "Avanti Sensor 1 (82703) | EMG 1 (mV)",
    "Avanti Sensor 2 (82529) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 2 (mV)",
]

N_CHANNELS = len(EMG_COLUMNS)
N_CLASSES = 2
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
RANDOM_SEED = 42
DEVICE = "cpu"

# Winning config from previous grid search.
CLASSIFIER_DROP = 0.3
CONV_DROP = 0.2

# Desired approximate downsampled length used to determine downsample factor.
# The actual final sequence length will be computed from the data.
TARGET_LEN_FOR_DOWNSAMPLE = 4200

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FIG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── DATA ────────────────────────────

def load_and_prepare():
    df = pl.read_parquet(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    missing_cols = [c for c in EMG_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing EMG columns: {missing_cols}")

    required_cols = {"run_id", "is_robot", "task_instance", "sample_idx"}
    missing_required = required_cols.difference(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    runs = {}
    for run_id in df["run_id"].unique().sort().to_list():
        run_df = df.filter(pl.col("run_id") == run_id)
        labels = run_df["is_robot"].unique().sort().to_list()

        if len(labels) != 1:
            raise ValueError(
                f"Run {run_id} has multiple is_robot labels: {labels}. "
                "This script assumes one label per run."
            )

        label = int(labels[0])
        n_instances = run_df["task_instance"].n_unique()
        runs[run_id] = {
            "df": run_df,
            "label": label,
            "n_instances": n_instances,
        }
        print(f"  {run_id}: label={label}, {n_instances} instances")

    n_robot = sum(1 for r in runs.values() if r["label"] == 1)
    n_norobot = sum(1 for r in runs.values() if r["label"] == 0)
    print(f"\nTotal: {len(runs)} runs ({n_robot} robot, {n_norobot} norobot)")
    print(f"Task instances: {df['task_instance'].n_unique()}")

    return df, runs


def compute_downsample_factor(runs, target_len=TARGET_LEN_FOR_DOWNSAMPLE):
    first_run = list(runs.values())[0]
    inst = first_run["df"]["task_instance"].unique()[0]
    sample_df = first_run["df"].filter(pl.col("task_instance") == inst).sort("sample_idx")
    raw_len = sample_df.select(EMG_COLUMNS).to_numpy().shape[0]

    factor = max(1, raw_len // target_len)
    actual_len = raw_len // factor

    print(f"  Data: {raw_len} samples -> downsample {factor}x -> {actual_len} timesteps")
    return factor, actual_len


def preprocess_window(window_df, downsample_factor, target_len):
    """Sort, downsample, pad/truncate, and return a (C, T) EMG window."""
    window_df = window_df.sort("sample_idx")
    raw = window_df.select(EMG_COLUMNS).to_numpy()
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    ds = raw[::downsample_factor]

    if ds.shape[0] >= target_len:
        ds = ds[:target_len]
    else:
        pad = target_len - ds.shape[0]
        ds = np.pad(ds, ((0, pad), (0, 0)), mode="constant")

    return ds.T.astype(np.float32)  # (C, T)


def extract_windows_from_run(run_df, downsample_factor, target_len):
    """Return one EMG window per task_instance."""
    windows = []
    instances = run_df["task_instance"].unique().sort().to_list()

    for inst in instances:
        inst_df = run_df.filter(pl.col("task_instance") == inst)
        windows.append(preprocess_window(inst_df, downsample_factor, target_len))

    return windows, instances


def compute_normalization_stats(windows):
    """Compute per-channel normalization stats from a list of (C, T) windows."""
    arr = np.array(windows, dtype=np.float32)  # (N, C, T)
    mean = arr.mean(axis=(0, 2), keepdims=True).astype(np.float32)  # (1, C, 1)
    std = arr.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    return mean, std


def normalize_windows(windows, mean, std):
    arr = np.array(windows, dtype=np.float32)
    return (arr - mean) / std


class EMGDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def build_fold_data(runs, held_out_run, downsample_factor, target_len):
    """Build train/test data for one LOOCV fold.

    Normalization stats are computed from the training runs only.
    The held-out run is never used to compute mean/std.
    """
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []
    test_instances = []

    for run_id, info in runs.items():
        run_windows, run_instances = extract_windows_from_run(
            info["df"],
            downsample_factor,
            target_len,
        )
        run_labels = [info["label"]] * len(run_windows)

        if run_id == held_out_run:
            test_windows.extend(run_windows)
            test_labels.extend(run_labels)
            test_instances.extend(run_instances)
        else:
            train_windows.extend(run_windows)
            train_labels.extend(run_labels)

    train_mean, train_std = compute_normalization_stats(train_windows)
    train_arr = normalize_windows(train_windows, train_mean, train_std)
    test_arr = normalize_windows(test_windows, train_mean, train_std)

    print(
        f"  Train: {len(train_labels)} windows "
        f"(robot={sum(train_labels)}, norobot={len(train_labels) - sum(train_labels)})"
    )
    print(
        f"  Test:  {len(test_labels)} windows "
        f"(robot={sum(test_labels)}, norobot={len(test_labels) - sum(test_labels)})"
    )

    return (
        EMGDataset(train_arr, train_labels),
        EMGDataset(test_arr, test_labels),
        train_mean,
        train_std,
        test_instances,
    )


def build_all_run_data(runs, downsample_factor, target_len):
    """Build final all-runs training data and final normalization stats."""
    windows, labels = [], []

    for _, info in runs.items():
        run_windows, _ = extract_windows_from_run(
            info["df"],
            downsample_factor,
            target_len,
        )
        windows.extend(run_windows)
        labels.extend([info["label"]] * len(run_windows))

    mean, std = compute_normalization_stats(windows)
    normalized = normalize_windows(windows, mean, std)

    print(
        f"  Final train: {len(labels)} windows "
        f"(robot={sum(labels)}, norobot={len(labels) - sum(labels)})"
    )

    return EMGDataset(normalized, labels), mean, std


# ──────────────────────────── MODEL ────────────────────────────

class EMGClassifier1D(nn.Module):
    def __init__(
        self,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        classifier_drop=CLASSIFIER_DROP,
        conv_drop=CONV_DROP,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(classifier_drop),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────── TRAINING / EVALUATION ────────────────────────────

def train_fixed_epochs(model, train_loader, num_epochs=NUM_EPOCHS):
    """Train without looking at the held-out test fold.

    This avoids test-fold leakage. If you want early stopping, use a separate
    validation run selected only from the training runs, not the held-out run.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:3d} | train_loss={train_loss:.4f}")

    model.cpu()
    return model, history


def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
    }


def safe_auc(labels, probs_robot):
    labels = np.array(labels)
    probs_robot = np.array(probs_robot)
    valid = ~np.isnan(probs_robot)

    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return 0.5

    fpr, tpr, _ = roc_curve(labels[valid], probs_robot[valid])
    return auc(fpr, tpr)


def binary_metrics(labels, preds, probs=None):
    labels = np.array(labels)
    preds = np.array(preds)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision_robot": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_robot": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_robot": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_norobot": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_norobot": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1_norobot": f1_score(labels, preds, pos_label=0, zero_division=0),
    }

    if probs is not None:
        probs = np.array(probs)
        metrics["roc_auc"] = safe_auc(labels, probs[:, 1])

    return metrics


# ──────────────────────────── PLOTS ────────────────────────────

def plot_confusion_matrix(all_labels, all_preds, save_path, title):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No-Robot", "Robot"],
        yticklabels=["No-Robot", "Robot"],
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
                fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(all_labels, all_probs, save_path, title):
    robot_probs = all_probs[:, 1]
    roc_auc = safe_auc(all_labels, robot_probs)

    valid = ~np.isnan(robot_probs)
    if valid.sum() < 2 or len(np.unique(np.array(all_labels)[valid])) < 2:
        return roc_auc

    fpr, tpr, _ = roc_curve(np.array(all_labels)[valid], robot_probs[valid])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return roc_auc


def plot_fold_accuracy(fold_accs, fold_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(fold_accs) + 1), fold_accs)
    ax.axhline(
        y=np.mean(fold_accs),
        linestyle="--",
        label=f"Mean: {np.mean(fold_accs):.3f}",
    )
    ax.set_xlabel("Fold / held-out run")
    ax.set_ylabel("Window-level accuracy")
    ax.set_title("Per-Fold Window-Level Accuracy")
    ax.set_xticks(range(1, len(fold_accs) + 1))
    ax.set_xticklabels(fold_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(all_histories, save_path):
    max_epochs = max(len(h["train_loss"]) for h in all_histories)
    train_losses = np.full((len(all_histories), max_epochs), np.nan)

    for i, h in enumerate(all_histories):
        n = len(h["train_loss"])
        train_losses[i, :n] = h["train_loss"]

    epochs = np.arange(1, max_epochs + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, np.nanmean(train_losses, axis=0), label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Mean Training Loss Across LOOCV Folds")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────── MAIN ────────────────────────────

def main():
    print("=" * 70)
    print("EMG 1D CNN — LOOCV Evaluation + Final Production Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(
        f"Config: conv_drop={CONV_DROP}, classifier_drop={CLASSIFIER_DROP}, "
        f"weight_decay={WEIGHT_DECAY}, lr={LR}, epochs={NUM_EPOCHS}"
    )

    df, runs = load_and_prepare()
    run_ids = sorted(runs.keys())

    downsample_factor, actual_seq_len = compute_downsample_factor(
        runs,
        target_len=TARGET_LEN_FOR_DOWNSAMPLE,
    )

    # ───────────────────── LOOCV EVALUATION ─────────────────────

    all_fold_metrics = []
    all_labels_agg, all_preds_agg, all_probs_agg = [], [], []
    all_histories = []
    fold_accuracies = []
    fold_names = []
    per_window_rows = []
    per_run_rows = []

    print(f"\n{'=' * 70}")
    print(f"Starting LOOCV evaluation: {len(run_ids)} folds")
    print(f"{'=' * 70}")

    for fold_idx, held_out in enumerate(run_ids):
        print(f"\n--- Fold {fold_idx + 1}/{len(run_ids)}: held out '{held_out}' ---")

        train_ds, test_ds, fold_mean, fold_std, test_instances = build_fold_data(
            runs,
            held_out,
            downsample_factor,
            actual_seq_len,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = EMGClassifier1D(n_channels=N_CHANNELS, n_classes=N_CLASSES)
        model, history = train_fixed_epochs(model, train_loader, num_epochs=NUM_EPOCHS)
        result = evaluate_model(model, test_loader)

        fold_metrics = binary_metrics(result["labels"], result["preds"], result["probs"])
        all_fold_metrics.append(fold_metrics)
        fold_accuracies.append(fold_metrics["accuracy"])
        fold_names.append(held_out)
        all_histories.append(history)

        all_labels_agg.extend(result["labels"])
        all_preds_agg.extend(result["preds"])
        all_probs_agg.extend(result["probs"])

        # Per-window predictions.
        for w in range(len(result["labels"])):
            per_window_rows.append({
                "fold": fold_idx + 1,
                "held_out_run": held_out,
                "task_instance": test_instances[w],
                "true_label": int(result["labels"][w]),
                "pred_label": int(result["preds"][w]),
                "prob_norobot": float(result["probs"][w][0]),
                "prob_robot": float(result["probs"][w][1]),
                "correct": int(result["labels"][w]) == int(result["preds"][w]),
            })

        # Run-level prediction by averaging window probabilities within the held-out run.
        run_true = int(result["labels"][0])
        run_prob_robot = float(result["probs"][:, 1].mean())
        run_prob_norobot = 1.0 - run_prob_robot
        run_pred = int(run_prob_robot >= 0.5)

        per_run_rows.append({
            "fold": fold_idx + 1,
            "held_out_run": held_out,
            "true_label": run_true,
            "pred_label": run_pred,
            "prob_norobot_mean": run_prob_norobot,
            "prob_robot_mean": run_prob_robot,
            "n_windows": len(result["labels"]),
            "window_accuracy": float(fold_metrics["accuracy"]),
            "correct": run_true == run_pred,
        })

        print(
            f"  Window accuracy: {fold_metrics['accuracy']:.3f} | "
            f"F1_robot: {fold_metrics['f1_robot']:.3f} | "
            f"Run pred: true={run_true}, pred={run_pred}, "
            f"prob_robot_mean={run_prob_robot:.3f}"
        )

    # Aggregate out-of-fold window-level metrics.
    all_labels_agg = np.array(all_labels_agg)
    all_preds_agg = np.array(all_preds_agg)
    all_probs_agg = np.array(all_probs_agg)

    aggregate_window = binary_metrics(all_labels_agg, all_preds_agg, all_probs_agg)
    aggregate_window["mean_fold_accuracy"] = float(np.mean(fold_accuracies))
    aggregate_window["std_fold_accuracy"] = (
        float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0
    )

    # Aggregate out-of-fold run-level metrics.
    run_labels = np.array([r["true_label"] for r in per_run_rows])
    run_preds = np.array([r["pred_label"] for r in per_run_rows])
    run_probs = np.array([[r["prob_norobot_mean"], r["prob_robot_mean"]] for r in per_run_rows])
    aggregate_run = binary_metrics(run_labels, run_preds, run_probs)

    print(f"\n{'=' * 70}")
    print("LOOCV AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    print("\nWindow-level out-of-fold metrics:")
    for k, v in aggregate_window.items():
        print(f"  {k}: {v:.4f}")

    print("\nRun-level out-of-fold metrics:")
    for k, v in aggregate_run.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nPer-fold window accuracies: {[f'{a:.3f}' for a in fold_accuracies]}")

    # ───────────────────── SAVE LOOCV OUTPUTS ─────────────────────

    print(f"\n{'=' * 70}")
    print("SAVING LOOCV OUTPUTS")
    print(f"{'=' * 70}")

    plot_fold_accuracy(fold_accuracies, fold_names, FIG_DIR / "fold_accuracy.png")
    plot_confusion_matrix(
        all_labels_agg,
        all_preds_agg,
        FIG_DIR / "confusion_matrix_window_level.png",
        "Window-Level Confusion Matrix — LOOCV",
    )
    plot_roc(
        all_labels_agg,
        all_probs_agg,
        FIG_DIR / "roc_curve_window_level.png",
        "Window-Level ROC Curve — LOOCV",
    )
    plot_confusion_matrix(
        run_labels,
        run_preds,
        FIG_DIR / "confusion_matrix_run_level.png",
        "Run-Level Confusion Matrix — LOOCV",
    )
    plot_roc(
        run_labels,
        run_probs,
        FIG_DIR / "roc_curve_run_level.png",
        "Run-Level ROC Curve — LOOCV",
    )
    plot_training_curves(all_histories, FIG_DIR / "training_curves.png")

    print(f"  Saved figures to {FIG_DIR}")

    # Per-window predictions CSV.
    pw_path = EXPORT_DIR / "predictions_window_level_loocv.csv"
    pw_fields = [
        "fold",
        "held_out_run",
        "task_instance",
        "true_label",
        "pred_label",
        "prob_norobot",
        "prob_robot",
        "correct",
    ]
    with open(pw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pw_fields)
        writer.writeheader()
        writer.writerows(per_window_rows)
    print(f"  Saved: {pw_path} ({len(per_window_rows)} windows)")

    # Per-run predictions CSV.
    pr_path = EXPORT_DIR / "predictions_run_level_loocv.csv"
    pr_fields = [
        "fold",
        "held_out_run",
        "true_label",
        "pred_label",
        "prob_norobot_mean",
        "prob_robot_mean",
        "n_windows",
        "window_accuracy",
        "correct",
    ]
    with open(pr_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pr_fields)
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"  Saved: {pr_path} ({len(per_run_rows)} runs)")

    # Metrics CSV.
    metrics_path = EXPORT_DIR / "metrics_loocv.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["date", datetime.now().isoformat()])
        writer.writerow(["model", "EMGClassifier1D"])
        writer.writerow(["evaluation", "leave-one-run-out CV"])
        writer.writerow(["n_channels", N_CHANNELS])
        writer.writerow(["emg_columns", json.dumps(EMG_COLUMNS)])
        writer.writerow(["downsample_factor", downsample_factor])
        writer.writerow(["seq_len", actual_seq_len])
        writer.writerow(["n_folds", len(run_ids)])
        writer.writerow(["num_epochs", NUM_EPOCHS])
        writer.writerow(["lr", LR])
        writer.writerow(["weight_decay", WEIGHT_DECAY])
        writer.writerow(["classifier_drop", CLASSIFIER_DROP])
        writer.writerow(["conv_drop", CONV_DROP])
        writer.writerow([])
        writer.writerow(["--- Window-Level Aggregate ---", ""])
        for k, v in aggregate_window.items():
            writer.writerow([k, f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Run-Level Aggregate ---", ""])
        for k, v in aggregate_run.items():
            writer.writerow([k, f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Per-Fold Window-Level Metrics ---", ""])
        for i, (m, rid) in enumerate(zip(all_fold_metrics, run_ids)):
            writer.writerow([f"fold_{i + 1}_run", rid])
            for k, v in m.items():
                writer.writerow([f"fold_{i + 1}_{k}", f"{v:.4f}"])
    print(f"  Saved: {metrics_path}")

    # ───────────────────── FINAL PRODUCTION TRAINING ─────────────────────

    print(f"\n{'=' * 70}")
    print("TRAINING FINAL PRODUCTION MODEL ON ALL RUNS")
    print(f"{'=' * 70}")

    final_train_ds, final_mean, final_std = build_all_run_data(
        runs,
        downsample_factor,
        actual_seq_len,
    )
    final_train_loader = DataLoader(final_train_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model = EMGClassifier1D(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    final_model, final_history = train_fixed_epochs(
        final_model,
        final_train_loader,
        num_epochs=NUM_EPOCHS,
    )

    production_config = {
        "model": "EMGClassifier1D",
        "variant": "ds5036_drop_conv",
        "data_path": str(DATA_PATH),
        "emg_columns": EMG_COLUMNS,
        "n_channels": N_CHANNELS,
        "n_classes": N_CLASSES,
        "downsample_factor": downsample_factor,
        "seq_len": actual_seq_len,
        "target_len_for_downsample": TARGET_LEN_FOR_DOWNSAMPLE,
        "classifier_drop": CLASSIFIER_DROP,
        "conv_drop": CONV_DROP,
        "weight_decay": WEIGHT_DECAY,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "device": DEVICE,
        "run_ids_used_for_final_training": run_ids,
        "loocv_window_metrics": {k: round(float(v), 4) for k, v in aggregate_window.items()},
        "loocv_run_metrics": {k: round(float(v), 4) for k, v in aggregate_run.items()},
    }

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "normalization_mean": final_mean.astype(np.float32),
        "normalization_std": final_std.astype(np.float32),
        "config": production_config,
    }

    model_path = EXPORT_DIR / "model_final_all_runs.pt"
    torch.save(checkpoint, model_path)
    print(f"  Saved production checkpoint: {model_path}")

    config_path = EXPORT_DIR / "config_final_all_runs.json"
    with open(config_path, "w") as f:
        json.dump(production_config, f, indent=2)
    print(f"  Saved production config: {config_path}")

    norm_path = EXPORT_DIR / "normalization_final_all_runs.npz"
    np.savez(
        norm_path,
        mean=final_mean.astype(np.float32),
        std=final_std.astype(np.float32),
        emg_columns=np.array(EMG_COLUMNS),
        downsample_factor=np.array([downsample_factor]),
        seq_len=np.array([actual_seq_len]),
    )
    print(f"  Saved normalization/downsampling stats: {norm_path}")

    print(f"\n{'=' * 70}")
    print("DONE — LOOCV evaluated; final all-runs production model exported")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
