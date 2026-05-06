"""
fNIRS 1D CNN — 10-Class Task Classification
============================================

Goal:
  Predict task identity from fNIRS windows, not robot vs no-robot.

Dataset structure assumed:
  - Multiple runs.
  - Each run contains task windows / task_instance values.
  - Each task_instance has one task label.
  - There are 10 task classes, each repeated multiple times per run.

This script does TWO separate things:

1. Leave-one-run-out cross-validation (LOOCV)
   - Hold out one full run at a time.
   - Train only on the remaining runs.
   - Normalize using training runs only.
   - Do NOT use the held-out run for early stopping, LR scheduling, or model selection.
   - Save per-window predictions, aggregate metrics, per-fold metrics, and plots.

2. Final production training
   - After LOOCV is complete, train one final model on ALL runs.
   - Compute final normalization stats from ALL runs.
   - Save the final all-runs model, normalization stats, task mapping, and config.

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
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ──────────────────────────── CONFIG ────────────────────────────

DATA_PATH = Path("./data/processed/combined/data_packet/fnirs_full.parquet")
FIG_DIR = Path("./machine_learning/fNIRS_pertask/figures")
EXPORT_DIR = Path("./machine_learning/fNIRS_pertask/export")

# Change this if your task label column has a different name.
# Common possibilities: "task", "task_id", "task_name", "task_label".
TASK_COLUMN = "task"

# The grouping column used for leave-one-run-out validation.
RUN_COLUMN = "run_id"

# Window identifier column. Each task_instance should correspond to one task window.
INSTANCE_COLUMN = "task_instance"

# Time column inside each fNIRS window.
TIME_COLUMN = "time_index"

# All fNIRS channels: 26 hbo + 26 hbr = 52 total.
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

N_CHANNELS = len(FNIRS_COLUMNS)
BATCH_SIZE = 8
RANDOM_SEED = 42
DEVICE = "cpu"

# Best-guess starting settings for 10-class task classification.
LR = 5e-4
WEIGHT_DECAY = 5e-3
NUM_EPOCHS = 75
CONV_DROP = 0.10
CLASSIFIER_DROP = 0.50
LABEL_SMOOTHING = 0.05

# If the model underfits badly, try:
#   WEIGHT_DECAY = 1e-3
#   CLASSIFIER_DROP = 0.30
#   LABEL_SMOOTHING = 0.0
#
# If the model overfits or has very unstable folds, try:
#   WEIGHT_DECAY = 1e-2
#   CONV_DROP = 0.20
#   CLASSIFIER_DROP = 0.60


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FIG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── DATA ────────────────────────────

def load_dataframe():
    df = pl.read_parquet(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    required_cols = {RUN_COLUMN, INSTANCE_COLUMN, TIME_COLUMN, TASK_COLUMN}
    missing_required = required_cols.difference(df.columns)
    if missing_required:
        raise ValueError(
            f"Missing required columns: {sorted(missing_required)}. "
            f"If your task column is named differently, change TASK_COLUMN near the top."
        )

    missing_fnirs = [c for c in FNIRS_COLUMNS if c not in df.columns]
    if missing_fnirs:
        raise ValueError(f"Missing fNIRS columns: {missing_fnirs}")

    return df


def make_task_mapping(df):
    """Create stable task label mappings."""
    task_values = df[TASK_COLUMN].unique().sort().to_list()
    task_to_idx = {str(task): i for i, task in enumerate(task_values)}
    idx_to_task = {i: str(task) for task, i in task_to_idx.items()}

    print("\nTask mapping:")
    for task, idx in task_to_idx.items():
        print(f"  {idx}: {task}")

    return task_to_idx, idx_to_task


def load_and_prepare():
    df = load_dataframe()
    task_to_idx, idx_to_task = make_task_mapping(df)
    n_classes = len(task_to_idx)

    if n_classes != 10:
        print(f"\nWARNING: Expected 10 task classes, found {n_classes}.")
        print("The script will still run using the detected number of classes.")

    runs = {}
    for run_id in df[RUN_COLUMN].unique().sort().to_list():
        run_df = df.filter(pl.col(RUN_COLUMN) == run_id)
        n_instances = run_df[INSTANCE_COLUMN].n_unique()
        task_counts = (
            run_df
            .select([INSTANCE_COLUMN, TASK_COLUMN])
            .unique()
            .group_by(TASK_COLUMN)
            .len()
            .sort(TASK_COLUMN)
        )

        runs[run_id] = {
            "df": run_df,
            "n_instances": n_instances,
        }

        print(f"  {run_id}: {n_instances} instances")
        print(task_counts)

    print(f"\nTotal runs: {len(runs)}")
    print(f"Total task instances/windows: {df[INSTANCE_COLUMN].n_unique()}")
    print(f"Detected classes: {n_classes}")

    return df, runs, task_to_idx, idx_to_task, n_classes


def preprocess_window(window_df):
    """Extract fNIRS channels, sort by time, and return a (C, T) array."""
    window_df = window_df.sort(TIME_COLUMN)
    raw = window_df.select(FNIRS_COLUMNS).to_numpy()  # (T, C)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw.T.astype(np.float32)  # (C, T)


def get_window_task_label(window_df, task_to_idx):
    labels = window_df[TASK_COLUMN].unique().to_list()
    if len(labels) != 1:
        raise ValueError(
            f"A single {INSTANCE_COLUMN} has multiple task labels: {labels}. "
            "Each task window must have exactly one task label."
        )
    task_value = str(labels[0])
    return task_to_idx[task_value], task_value


def extract_windows_from_run(run_df, task_to_idx):
    """Return one fNIRS window and one task label per task_instance."""
    windows = []
    labels = []
    task_values = []
    instances = run_df[INSTANCE_COLUMN].unique().sort().to_list()

    for inst in instances:
        inst_df = run_df.filter(pl.col(INSTANCE_COLUMN) == inst)
        label_idx, task_value = get_window_task_label(inst_df, task_to_idx)
        windows.append(preprocess_window(inst_df))
        labels.append(label_idx)
        task_values.append(task_value)

    return windows, labels, task_values, instances


def compute_normalization_stats(windows):
    """Compute per-channel normalization stats from a list of (C, T) windows."""
    arr = np.array(windows, dtype=np.float32)  # (N, C, T)
    mean = arr.mean(axis=(0, 2), keepdims=True).astype(np.float32)  # (1, C, 1)
    std = arr.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
    return mean, std


def normalize_windows(windows, mean, std):
    arr = np.array(windows, dtype=np.float32)
    return (arr - mean) / std


class FNIRSTaskDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def build_fold_data(runs, held_out_run, task_to_idx):
    """Build train/test data for one LOOCV fold.

    Normalization stats are computed from the training runs only.
    The held-out run is never used to compute mean/std.
    """
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []
    test_task_values = []
    test_instances = []

    for run_id, info in runs.items():
        run_windows, run_labels, run_task_values, run_instances = extract_windows_from_run(
            info["df"],
            task_to_idx,
        )

        if run_id == held_out_run:
            test_windows.extend(run_windows)
            test_labels.extend(run_labels)
            test_task_values.extend(run_task_values)
            test_instances.extend(run_instances)
        else:
            train_windows.extend(run_windows)
            train_labels.extend(run_labels)

    train_mean, train_std = compute_normalization_stats(train_windows)
    train_arr = normalize_windows(train_windows, train_mean, train_std)
    test_arr = normalize_windows(test_windows, train_mean, train_std)

    print(f"  Train: {len(train_labels)} windows")
    print(f"  Test:  {len(test_labels)} windows")

    return (
        FNIRSTaskDataset(train_arr, train_labels),
        FNIRSTaskDataset(test_arr, test_labels),
        train_mean,
        train_std,
        test_instances,
        test_task_values,
    )


def build_all_run_data(runs, task_to_idx):
    """Build final all-runs training data and final normalization stats."""
    windows, labels = [], []

    for _, info in runs.items():
        run_windows, run_labels, _, _ = extract_windows_from_run(info["df"], task_to_idx)
        windows.extend(run_windows)
        labels.extend(run_labels)

    mean, std = compute_normalization_stats(windows)
    normalized = normalize_windows(windows, mean, std)

    print(f"  Final train: {len(labels)} windows")
    return FNIRSTaskDataset(normalized, labels), mean, std


# ──────────────────────────── MODEL ────────────────────────────

class FNIRSTaskCNN(nn.Module):
    """fNIRS 1D CNN for task classification.

    Input shape:
        (batch, 52, 201)

    Output shape:
        (batch, n_classes)
    """

    def __init__(
        self,
        n_channels=N_CHANNELS,
        n_classes=10,
        conv_drop=CONV_DROP,
        classifier_drop=CLASSIFIER_DROP,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # 201 -> 100
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),

            # 100 -> 50
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),

            # 50 -> 25
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
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
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
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


def multiclass_metrics(labels, preds, n_classes):
    labels = np.array(labels)
    preds = np.array(preds)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(n_classes)),
        zero_division=0,
    )

    per_class = []
    for i in range(n_classes):
        per_class.append({
            "class_idx": i,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    return metrics, per_class


# ──────────────────────────── PLOTS ────────────────────────────

def plot_confusion_matrix(labels, preds, idx_to_task, save_path, title):
    n_classes = len(idx_to_task)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))

    fig_size = max(7, n_classes * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    tick_labels = [idx_to_task[i] for i in range(n_classes)]
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ylabel="True task",
        xlabel="Predicted task",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_accuracy(fold_accs, fold_names, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, len(fold_accs) + 1), fold_accs)
    ax.axhline(
        y=np.mean(fold_accs),
        linestyle="--",
        label=f"Mean: {np.mean(fold_accs):.3f}",
    )
    ax.set_xlabel("Fold / held-out run")
    ax.set_ylabel("Task accuracy")
    ax.set_title("Per-Fold Task Accuracy")
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
    print("=" * 80)
    print("fNIRS 1D CNN — 10-Class Task LOOCV + Final All-Runs Training")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(
        f"Channels: {N_CHANNELS} "
        f"({len([c for c in FNIRS_COLUMNS if c.endswith('_hbo')])} hbo + "
        f"{len([c for c in FNIRS_COLUMNS if c.endswith('_hbr')])} hbr)"
    )
    print(
        f"Config: conv_drop={CONV_DROP}, classifier_drop={CLASSIFIER_DROP}, "
        f"weight_decay={WEIGHT_DECAY}, lr={LR}, epochs={NUM_EPOCHS}, "
        f"label_smoothing={LABEL_SMOOTHING}"
    )
    print(f"Task column: {TASK_COLUMN}")

    df, runs, task_to_idx, idx_to_task, n_classes = load_and_prepare()
    run_ids = sorted(runs.keys())

    sample_df = list(runs.values())[0]["df"]
    inst = sample_df[INSTANCE_COLUMN].unique()[0]
    seq_len = sample_df.filter(pl.col(INSTANCE_COLUMN) == inst).shape[0]
    print(f"\nSequence length: {seq_len} timesteps")

    # ───────────────────── LOOCV EVALUATION ─────────────────────

    all_fold_metrics = []
    all_fold_per_class = []
    all_labels_agg, all_preds_agg, all_probs_agg = [], [], []
    all_histories = []
    fold_accuracies = []
    fold_names = []
    per_window_rows = []

    print(f"\n{'=' * 80}")
    print(f"Starting LOOCV evaluation: {len(run_ids)} folds")
    print(f"{'=' * 80}")

    for fold_idx, held_out in enumerate(run_ids):
        print(f"\n--- Fold {fold_idx + 1}/{len(run_ids)}: held out '{held_out}' ---")

        train_ds, test_ds, fold_mean, fold_std, test_instances, test_task_values = build_fold_data(
            runs,
            held_out,
            task_to_idx,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = FNIRSTaskCNN(n_channels=N_CHANNELS, n_classes=n_classes)
        model, history = train_fixed_epochs(model, train_loader, num_epochs=NUM_EPOCHS)
        result = evaluate_model(model, test_loader)

        fold_metrics, fold_per_class = multiclass_metrics(
            result["labels"],
            result["preds"],
            n_classes,
        )
        all_fold_metrics.append(fold_metrics)
        all_fold_per_class.append(fold_per_class)
        fold_accuracies.append(fold_metrics["accuracy"])
        fold_names.append(held_out)
        all_histories.append(history)

        all_labels_agg.extend(result["labels"])
        all_preds_agg.extend(result["preds"])
        all_probs_agg.extend(result["probs"])

        # Per-window predictions with per-class probabilities.
        for w in range(len(result["labels"])):
            true_idx = int(result["labels"][w])
            pred_idx = int(result["preds"][w])
            row = {
                "fold": fold_idx + 1,
                "held_out_run": held_out,
                "task_instance": test_instances[w],
                "true_task": idx_to_task[true_idx],
                "true_label": true_idx,
                "pred_task": idx_to_task[pred_idx],
                "pred_label": pred_idx,
                "correct": true_idx == pred_idx,
                "pred_confidence": float(result["probs"][w][pred_idx]),
            }
            for class_idx in range(n_classes):
                row[f"prob_{idx_to_task[class_idx]}"] = float(result["probs"][w][class_idx])
            per_window_rows.append(row)

        print(
            f"  Accuracy: {fold_metrics['accuracy']:.3f} | "
            f"Balanced acc: {fold_metrics['balanced_accuracy']:.3f} | "
            f"Macro F1: {fold_metrics['macro_f1']:.3f}"
        )

    # Aggregate out-of-fold metrics.
    all_labels_agg = np.array(all_labels_agg)
    all_preds_agg = np.array(all_preds_agg)
    all_probs_agg = np.array(all_probs_agg)

    aggregate_metrics, aggregate_per_class = multiclass_metrics(
        all_labels_agg,
        all_preds_agg,
        n_classes,
    )
    aggregate_metrics["mean_fold_accuracy"] = float(np.mean(fold_accuracies))
    aggregate_metrics["std_fold_accuracy"] = (
        float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0
    )

    report = classification_report(
        all_labels_agg,
        all_preds_agg,
        labels=list(range(n_classes)),
        target_names=[idx_to_task[i] for i in range(n_classes)],
        zero_division=0,
    )

    print(f"\n{'=' * 80}")
    print("LOOCV AGGREGATE RESULTS")
    print(f"{'=' * 80}")
    for k, v in aggregate_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nClassification report:")
    print(report)
    print(f"Per-fold accuracies: {[f'{a:.3f}' for a in fold_accuracies]}")

    # ───────────────────── SAVE LOOCV OUTPUTS ─────────────────────

    print(f"\n{'=' * 80}")
    print("SAVING LOOCV OUTPUTS")
    print(f"{'=' * 80}")

    plot_fold_accuracy(fold_accuracies, fold_names, FIG_DIR / "fold_accuracy.png")
    plot_confusion_matrix(
        all_labels_agg,
        all_preds_agg,
        idx_to_task,
        FIG_DIR / "confusion_matrix_task_level.png",
        "Task Confusion Matrix — LOOCV",
    )
    plot_training_curves(all_histories, FIG_DIR / "training_curves.png")
    print(f"  Saved figures to {FIG_DIR}")

    # Per-window predictions CSV.
    predictions_path = EXPORT_DIR / "predictions_task_level_loocv.csv"
    prediction_fields = list(per_window_rows[0].keys()) if per_window_rows else []
    with open(predictions_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(per_window_rows)
    print(f"  Saved: {predictions_path} ({len(per_window_rows)} windows)")

    # Metrics CSV.
    metrics_path = EXPORT_DIR / "metrics_loocv.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["date", datetime.now().isoformat()])
        writer.writerow(["model", "FNIRSTaskCNN"])
        writer.writerow(["evaluation", "leave-one-run-out CV"])
        writer.writerow(["task_column", TASK_COLUMN])
        writer.writerow(["n_channels", N_CHANNELS])
        writer.writerow(["channel_types", "26 hbo + 26 hbr"])
        writer.writerow(["seq_len", seq_len])
        writer.writerow(["sampling_rate_hz", 10])
        writer.writerow(["n_classes", n_classes])
        writer.writerow(["n_folds", len(run_ids)])
        writer.writerow(["num_epochs", NUM_EPOCHS])
        writer.writerow(["lr", LR])
        writer.writerow(["weight_decay", WEIGHT_DECAY])
        writer.writerow(["classifier_drop", CLASSIFIER_DROP])
        writer.writerow(["conv_drop", CONV_DROP])
        writer.writerow(["label_smoothing", LABEL_SMOOTHING])
        writer.writerow([])
        writer.writerow(["--- Aggregate ---", ""])
        for k, v in aggregate_metrics.items():
            writer.writerow([k, f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Per-Class Aggregate ---", ""])
        for pc in aggregate_per_class:
            class_name = idx_to_task[pc["class_idx"]]
            writer.writerow([f"class_{pc['class_idx']}_name", class_name])
            writer.writerow([f"class_{pc['class_idx']}_precision", f"{pc['precision']:.4f}"])
            writer.writerow([f"class_{pc['class_idx']}_recall", f"{pc['recall']:.4f}"])
            writer.writerow([f"class_{pc['class_idx']}_f1", f"{pc['f1']:.4f}"])
            writer.writerow([f"class_{pc['class_idx']}_support", pc["support"]])
        writer.writerow([])
        writer.writerow(["--- Per-Fold ---", ""])
        for i, (m, rid) in enumerate(zip(all_fold_metrics, run_ids)):
            writer.writerow([f"fold_{i + 1}_run", rid])
            for k, v in m.items():
                writer.writerow([f"fold_{i + 1}_{k}", f"{v:.4f}"])
    print(f"  Saved: {metrics_path}")

    # Classification report text.
    report_path = EXPORT_DIR / "classification_report_loocv.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Task mapping.
    mapping_path = EXPORT_DIR / "task_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({
            "task_column": TASK_COLUMN,
            "task_to_idx": task_to_idx,
            "idx_to_task": idx_to_task,
        }, f, indent=2)
    print(f"  Saved: {mapping_path}")

    # ───────────────────── FINAL PRODUCTION TRAINING ─────────────────────

    print(f"\n{'=' * 80}")
    print("TRAINING FINAL PRODUCTION MODEL ON ALL RUNS")
    print(f"{'=' * 80}")

    final_train_ds, final_mean, final_std = build_all_run_data(runs, task_to_idx)
    final_train_loader = DataLoader(final_train_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model = FNIRSTaskCNN(n_channels=N_CHANNELS, n_classes=n_classes)
    final_model, final_history = train_fixed_epochs(
        final_model,
        final_train_loader,
        num_epochs=NUM_EPOCHS,
    )

    production_config = {
        "model": "FNIRSTaskCNN",
        "task": "10-class task classification",
        "data_path": str(DATA_PATH),
        "task_column": TASK_COLUMN,
        "run_column": RUN_COLUMN,
        "instance_column": INSTANCE_COLUMN,
        "time_column": TIME_COLUMN,
        "fnirs_columns": FNIRS_COLUMNS,
        "n_channels": N_CHANNELS,
        "n_classes": n_classes,
        "task_to_idx": task_to_idx,
        "idx_to_task": idx_to_task,
        "channel_types": "26 hbo + 26 hbr",
        "seq_len": seq_len,
        "sampling_rate_hz": 10,
        "classifier_drop": CLASSIFIER_DROP,
        "conv_drop": CONV_DROP,
        "weight_decay": WEIGHT_DECAY,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "label_smoothing": LABEL_SMOOTHING,
        "device": DEVICE,
        "run_ids_used_for_final_training": run_ids,
        "loocv_metrics": {k: round(float(v), 4) for k, v in aggregate_metrics.items()},
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
        fnirs_columns=np.array(FNIRS_COLUMNS),
        seq_len=np.array([seq_len]),
    )
    print(f"  Saved normalization stats: {norm_path}")

    print(f"\n{'=' * 80}")
    print("DONE — LOOCV evaluated; final all-runs task model exported")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
