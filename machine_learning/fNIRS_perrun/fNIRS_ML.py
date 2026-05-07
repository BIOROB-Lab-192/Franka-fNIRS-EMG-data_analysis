"""
fNIRS 1D CNN — LOOCV Evaluation + Final All-Runs Production Model
==================================================================

Dataset structure:
  - multiple runs
  - each run contains multiple task windows / task_instance values
  - each run has a run-level label: is_robot

This script does two separate things:

1. LOOCV evaluation by run_id
   - Hold out one full run at a time.
   - Train only on the other runs.
   - Normalize using training runs only.
   - Do NOT use the held-out run for early stopping, LR scheduling, or model selection.
   - Use a fixed, predetermined cosine LR schedule.
   - Report window-level and run-level out-of-fold metrics.
   - Save run-level aggregation diagnostics: mean probability, median probability, majority vote.

2. Final production training
   - After LOOCV is complete, train one final model on all runs.
   - Compute final normalization stats from all runs.
   - Save final model weights, normalization stats, fNIRS metadata, and config.

Important:
  The LOOCV fold models are temporary evaluation models.
  The saved production model is trained on all available runs.
"""

import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)


# ──────────────────────────── CONFIG ────────────────────────────

DATA_PATH = Path("./data/processed/combined/data_packet/fnirs_full.parquet")
FIG_DIR = Path("./machine_learning/fNIRS_perrun/figures")
EXPORT_DIR = Path("./machine_learning/fNIRS_perrun/export")

# All fNIRS channels: 26 hbo + 26 hbr = 52 total.
FNIRS_COLUMNS = [
    "S1_D1_hbo",
    "S1_D2_hbo",
    "S1_D3_hbo",
    "S1_D8_hbo",
    "S2_D1_hbo",
    "S2_D3_hbo",
    "S2_D4_hbo",
    "S2_D9_hbo",
    "S3_D2_hbo",
    "S3_D3_hbo",
    "S3_D10_hbo",
    "S4_D3_hbo",
    "S4_D4_hbo",
    "S4_D11_hbo",
    "S5_D5_hbo",
    "S5_D6_hbo",
    "S5_D7_hbo",
    "S5_D12_hbo",
    "S6_D5_hbo",
    "S6_D7_hbo",
    "S6_D13_hbo",
    "S7_D6_hbo",
    "S7_D7_hbo",
    "S7_D14_hbo",
    "S8_D7_hbo",
    "S8_D15_hbo",
    "S1_D1_hbr",
    "S1_D2_hbr",
    "S1_D3_hbr",
    "S1_D8_hbr",
    "S2_D1_hbr",
    "S2_D3_hbr",
    "S2_D4_hbr",
    "S2_D9_hbr",
    "S3_D2_hbr",
    "S3_D3_hbr",
    "S3_D10_hbr",
    "S4_D3_hbr",
    "S4_D4_hbr",
    "S4_D11_hbr",
    "S5_D5_hbr",
    "S5_D6_hbr",
    "S5_D7_hbr",
    "S5_D12_hbr",
    "S6_D5_hbr",
    "S6_D7_hbr",
    "S6_D13_hbr",
    "S7_D6_hbr",
    "S7_D7_hbr",
    "S7_D14_hbr",
    "S8_D7_hbr",
    "S8_D15_hbr",
]

N_CHANNELS = len(FNIRS_COLUMNS)
N_CLASSES = 2
BATCH_SIZE = 8
RANDOM_SEED = 42
DEVICE = "cpu"

# Conservative fNIRS training settings.
LR = 5e-4
WEIGHT_DECAY = 5e-3
NUM_EPOCHS = 50
CLASSIFIER_DROP = 0.5
CONV_DROP = 0.1
LABEL_SMOOTHING = 0.05

# fNIRS metadata.
SAMPLING_RATE_HZ = 10
BANDPASS_DESCRIPTION = "Standard bandpass to remove heartbeat, then conversion to HbO/HbR. Exact cutoffs should be documented."

# If your windows are variable length, set this to an integer such as 201.
# If None, the script requires all task windows to have the same length.
TARGET_SEQ_LEN = None


random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

FIG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── DATA ────────────────────────────


def load_and_prepare():
    df = pl.read_parquet(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    missing_cols = [c for c in FNIRS_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing fNIRS columns: {missing_cols}")

    required_cols = {"run_id", "is_robot", "task_instance", "time_index"}
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
    n_run_instance_pairs = df.select(["run_id", "task_instance"]).unique().height

    print(f"\nTotal: {len(runs)} runs ({n_robot} robot, {n_norobot} norobot)")
    print(f"Run-task windows: {n_run_instance_pairs}")
    print(f"Unique task_instance values globally: {df['task_instance'].n_unique()}")

    return df, runs


def get_window_length_summary(runs):
    lengths = []

    for info in runs.values():
        run_df = info["df"]
        for inst in run_df["task_instance"].unique().sort().to_list():
            inst_df = run_df.filter(pl.col("task_instance") == inst)
            lengths.append(inst_df.height)

    lengths = np.array(lengths, dtype=int)

    print(
        "\nfNIRS window length summary: "
        f"min={lengths.min()}, "
        f"median={np.median(lengths):.1f}, "
        f"max={lengths.max()}, "
        f"unique={sorted(np.unique(lengths).tolist())}"
    )

    return lengths


def summarize_signal_quality(runs):
    rows = []

    for run_id, info in runs.items():
        run_df = info["df"]
        arr = run_df.select(FNIRS_COLUMNS).to_numpy()

        finite = np.isfinite(arr)
        finite_values = arr[finite]

        if finite_values.size == 0:
            value_mean = np.nan
            value_std = np.nan
            value_min = np.nan
            value_max = np.nan
        else:
            value_mean = float(np.mean(finite_values))
            value_std = float(np.std(finite_values))
            value_min = float(np.min(finite_values))
            value_max = float(np.max(finite_values))

        rows.append(
            {
                "run_id": run_id,
                "label": info["label"],
                "n_rows": arr.shape[0],
                "n_windows": info["n_instances"],
                "n_nan": int(np.isnan(arr).sum()),
                "n_posinf": int(np.isposinf(arr).sum()),
                "n_neginf": int(np.isneginf(arr).sum()),
                "value_mean": value_mean,
                "value_std": value_std,
                "value_min": value_min,
                "value_max": value_max,
            }
        )

    return rows


def save_signal_quality(qc_rows):
    qc_path = EXPORT_DIR / "signal_quality_by_run.csv"
    fields = [
        "run_id",
        "label",
        "n_rows",
        "n_windows",
        "n_nan",
        "n_posinf",
        "n_neginf",
        "value_mean",
        "value_std",
        "value_min",
        "value_max",
    ]

    with open(qc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(qc_rows)

    print(f"  Saved signal quality summary: {qc_path}")


def preprocess_window(window_df, target_len=None):
    """Extract fNIRS channels, sort by time, optionally pad/truncate, and return (C, T)."""
    window_df = window_df.sort("time_index")
    raw = window_df.select(FNIRS_COLUMNS).to_numpy()  # (T, C)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    if target_len is not None:
        if raw.shape[0] >= target_len:
            raw = raw[:target_len]
        else:
            pad = target_len - raw.shape[0]
            raw = np.pad(raw, ((0, pad), (0, 0)), mode="constant")

    return raw.T.astype(np.float32)  # (C, T)


def extract_windows_from_run(run_df, target_len=None):
    """Return one fNIRS window per task_instance."""
    windows = []
    instances = run_df["task_instance"].unique().sort().to_list()

    for inst in instances:
        inst_df = run_df.filter(pl.col("task_instance") == inst)
        windows.append(preprocess_window(inst_df, target_len=target_len))

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


class FNIRSDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def build_fold_data(runs, held_out_run, target_len=None):
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
            target_len=target_len,
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
        FNIRSDataset(train_arr, train_labels),
        FNIRSDataset(test_arr, test_labels),
        train_mean,
        train_std,
        test_instances,
    )


def build_all_run_data(runs, target_len=None):
    """Build final all-runs training data and final normalization stats."""
    windows, labels = [], []

    for _, info in runs.items():
        run_windows, _ = extract_windows_from_run(
            info["df"],
            target_len=target_len,
        )
        windows.extend(run_windows)
        labels.extend([info["label"]] * len(run_windows))

    mean, std = compute_normalization_stats(windows)
    normalized = normalize_windows(windows, mean, std)

    print(
        f"  Final train: {len(labels)} windows "
        f"(robot={sum(labels)}, norobot={len(labels) - sum(labels)})"
    )

    return FNIRSDataset(normalized, labels), mean, std


# ──────────────────────────── MODEL ────────────────────────────


class FNIRSClassifier1D(nn.Module):
    """Original-style 1D CNN for fNIRS data: input shape (batch, 52, 201)."""

    def __init__(
        self,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES,
        classifier_drop=CLASSIFIER_DROP,
        conv_drop=CONV_DROP,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # (52, 201) -> (32, 100)
            nn.Conv1d(n_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            # (32, 100) -> (64, 50)
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            # (64, 50) -> (128, 25)
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(conv_drop),
            nn.MaxPool1d(2),
            # (128, 25) -> (256, 1)
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
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



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────── TRAINING / EVALUATION ────────────────────────────


def train_fixed_epochs(model, train_loader, num_epochs=NUM_EPOCHS):
    """Train for a fixed number of epochs using a fixed LR schedule.

    No validation or held-out fold performance is used for model selection.
    The LR schedule is predetermined from num_epochs only.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=LR * 0.01,
    )

    history = {
        "train_loss": [],
        "lr": [],
    }

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

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["lr"].append(current_lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch + 1:3d} | "
                f"train_loss={train_loss:.4f} | "
                f"lr={current_lr:.6g}"
            )

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
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision_robot": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_robot": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_robot": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_norobot": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_norobot": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1_norobot": f1_score(labels, preds, pos_label=0, zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
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

    # Choose text color based on actual cell color brightness
    cmap = im.cmap
    norm = im.norm

    for i in range(2):
        for j in range(2):
            rgba = cmap(norm(cm[i, j]))
            r, g, b, _ = rgba

            # Perceived brightness; lower = darker background
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if brightness < 0.45 else "black"

            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color=text_color,
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


def plot_single_training_curve(history, save_path, title):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────── MAIN ────────────────────────────


def main():
    print("=" * 70)
    print("fNIRS 1D CNN — LOOCV Evaluation + Final All-Runs Production Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(
        f"Channels: {N_CHANNELS} "
        f"({len([c for c in FNIRS_COLUMNS if c.endswith('_hbo')])} hbo + "
        f"{len([c for c in FNIRS_COLUMNS if c.endswith('_hbr')])} hbr)"
    )
    print(
        f"Config: conv_drop={CONV_DROP}, classifier_drop={CLASSIFIER_DROP}, "
        f"weight_decay={WEIGHT_DECAY}, lr={LR}, epochs={NUM_EPOCHS}, "
        f"label_smoothing={LABEL_SMOOTHING}, batch_size={BATCH_SIZE}"
    )

    df, runs = load_and_prepare()
    run_ids = sorted(runs.keys())

    qc_rows = summarize_signal_quality(runs)
    save_signal_quality(qc_rows)

    lengths = get_window_length_summary(runs)
    if TARGET_SEQ_LEN is None:
        if len(np.unique(lengths)) != 1:
            raise ValueError(
                "fNIRS windows have variable lengths. Set TARGET_SEQ_LEN to an integer "
                "to enable pad/truncate before training."
            )
        seq_len = int(lengths[0])
        target_len_for_preprocess = None
    else:
        seq_len = int(TARGET_SEQ_LEN)
        target_len_for_preprocess = seq_len

    print(f"\nSequence length used by model: {seq_len} timesteps")

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
            target_len=target_len_for_preprocess,
        )

        g = torch.Generator()
        g.manual_seed(RANDOM_SEED + fold_idx)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=g,
        )
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = FNIRSClassifier1D(n_channels=N_CHANNELS, n_classes=N_CLASSES)
        print(f"  Model parameters: {count_parameters(model):,}")

        model, history = train_fixed_epochs(model, train_loader, num_epochs=NUM_EPOCHS)
        result = evaluate_model(model, test_loader)

        fold_metrics = binary_metrics(
            result["labels"],
            result["preds"],
            result["probs"],
        )
        all_fold_metrics.append(fold_metrics)
        fold_accuracies.append(fold_metrics["accuracy"])
        fold_names.append(held_out)
        all_histories.append(history)

        all_labels_agg.extend(result["labels"])
        all_preds_agg.extend(result["preds"])
        all_probs_agg.extend(result["probs"])

        # Per-window predictions.
        for w in range(len(result["labels"])):
            per_window_rows.append(
                {
                    "fold": fold_idx + 1,
                    "held_out_run": held_out,
                    "task_instance": test_instances[w],
                    "true_label": int(result["labels"][w]),
                    "pred_label": int(result["preds"][w]),
                    "prob_norobot": float(result["probs"][w][0]),
                    "prob_robot": float(result["probs"][w][1]),
                    "correct": int(result["labels"][w]) == int(result["preds"][w]),
                }
            )

        # Run-level prediction diagnostics.
        run_true = int(result["labels"][0])
        robot_probs = result["probs"][:, 1]
        window_preds = result["preds"]

        run_prob_robot_mean = float(robot_probs.mean())
        run_prob_robot_median = float(np.median(robot_probs))

        run_pred_mean_prob = int(run_prob_robot_mean >= 0.5)
        run_pred_median_prob = int(run_prob_robot_median >= 0.5)
        run_pred_majority = int(window_preds.mean() >= 0.5)

        # Default run-level aggregation: mean probability.
        run_pred = run_pred_mean_prob
        run_prob_robot = run_prob_robot_mean
        run_prob_norobot = 1.0 - run_prob_robot

        per_run_rows.append(
            {
                "fold": fold_idx + 1,
                "held_out_run": held_out,
                "true_label": run_true,

                # Original/default aggregation: mean probability.
                "pred_label": run_pred,
                "prob_norobot_mean": run_prob_norobot,
                "prob_robot_mean": run_prob_robot,
                "correct": run_true == run_pred,

                # Alternative aggregation diagnostics.
                "pred_label_mean_prob": run_pred_mean_prob,
                "pred_label_median_prob": run_pred_median_prob,
                "pred_label_majority": run_pred_majority,

                "prob_robot_median": run_prob_robot_median,
                "prob_robot_std": float(robot_probs.std(ddof=1)) if len(robot_probs) > 1 else 0.0,
                "prob_robot_min": float(robot_probs.min()),
                "prob_robot_max": float(robot_probs.max()),

                "correct_mean_prob": run_true == run_pred_mean_prob,
                "correct_median_prob": run_true == run_pred_median_prob,
                "correct_majority": run_true == run_pred_majority,

                "n_windows": len(result["labels"]),
                "window_accuracy": float(fold_metrics["accuracy"]),
            }
        )

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

    # Aggregate out-of-fold run-level metrics using default mean-probability aggregation.
    run_labels = np.array([r["true_label"] for r in per_run_rows])
    run_preds = np.array([r["pred_label"] for r in per_run_rows])
    run_probs = np.array(
        [[r["prob_norobot_mean"], r["prob_robot_mean"]] for r in per_run_rows]
    )
    aggregate_run = binary_metrics(run_labels, run_preds, run_probs)

    # Optional aggregate diagnostics for alternative run aggregation rules.
    run_preds_median = np.array([r["pred_label_median_prob"] for r in per_run_rows])
    run_preds_majority = np.array([r["pred_label_majority"] for r in per_run_rows])
    aggregate_run_median = binary_metrics(run_labels, run_preds_median, run_probs)
    aggregate_run_majority = binary_metrics(run_labels, run_preds_majority, run_probs)

    print(f"\n{'=' * 70}")
    print("LOOCV AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    print("\nWindow-level out-of-fold metrics:")
    for k, v in aggregate_window.items():
        print(f"  {k}: {v:.4f}")

    print("\nRun-level out-of-fold metrics, mean-probability aggregation:")
    for k, v in aggregate_run.items():
        print(f"  {k}: {v:.4f}")

    print("\nRun-level diagnostics, median-probability aggregation:")
    for k, v in aggregate_run_median.items():
        print(f"  {k}: {v:.4f}")

    print("\nRun-level diagnostics, majority-vote aggregation:")
    for k, v in aggregate_run_majority.items():
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

        # Original/default aggregation: mean probability.
        "pred_label",
        "prob_norobot_mean",
        "prob_robot_mean",
        "correct",

        # Alternative aggregation diagnostics.
        "pred_label_mean_prob",
        "pred_label_median_prob",
        "pred_label_majority",

        "prob_robot_median",
        "prob_robot_std",
        "prob_robot_min",
        "prob_robot_max",

        "correct_mean_prob",
        "correct_median_prob",
        "correct_majority",

        "n_windows",
        "window_accuracy",
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
        writer.writerow(["model", "FNIRSClassifier1D"])
        writer.writerow(["variant", "small_gn_16_32_64"])
        writer.writerow(["evaluation", "leave-one-run-out CV"])
        writer.writerow(["n_channels", N_CHANNELS])
        writer.writerow(["fnirs_columns", json.dumps(FNIRS_COLUMNS)])
        writer.writerow(["channel_types", "26 hbo + 26 hbr"])
        writer.writerow(["seq_len", seq_len])
        writer.writerow(["target_seq_len", TARGET_SEQ_LEN if TARGET_SEQ_LEN is not None else "none"])
        writer.writerow(["sampling_rate_hz", SAMPLING_RATE_HZ])
        writer.writerow(["bandpass_description", BANDPASS_DESCRIPTION])
        writer.writerow(["n_folds", len(run_ids)])
        writer.writerow(["num_epochs", NUM_EPOCHS])
        writer.writerow(["lr", LR])
        writer.writerow(["lr_schedule", "CosineAnnealingLR"])
        writer.writerow(["lr_schedule_t_max", NUM_EPOCHS])
        writer.writerow(["lr_schedule_eta_min", LR * 0.01])
        writer.writerow(["weight_decay", WEIGHT_DECAY])
        writer.writerow(["batch_size", BATCH_SIZE])
        writer.writerow(["classifier_drop", CLASSIFIER_DROP])
        writer.writerow(["conv_drop", CONV_DROP])
        writer.writerow(["label_smoothing", LABEL_SMOOTHING])
        writer.writerow([])
        writer.writerow(["--- Window-Level Aggregate ---", ""])
        for k, v in aggregate_window.items():
            writer.writerow([k, f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Run-Level Aggregate: Mean Probability ---", ""])
        for k, v in aggregate_run.items():
            writer.writerow([k, f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Run-Level Diagnostic: Median Probability ---", ""])
        for k, v in aggregate_run_median.items():
            writer.writerow([f"median_{k}", f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Run-Level Diagnostic: Majority Vote ---", ""])
        for k, v in aggregate_run_majority.items():
            writer.writerow([f"majority_{k}", f"{v:.4f}"])
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
        target_len=target_len_for_preprocess,
    )

    g_final = torch.Generator()
    g_final.manual_seed(RANDOM_SEED)

    final_train_loader = DataLoader(
        final_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g_final,
    )

    final_model = FNIRSClassifier1D(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    print(f"  Final model parameters: {count_parameters(final_model):,}")

    final_model, final_history = train_fixed_epochs(
        final_model,
        final_train_loader,
        num_epochs=NUM_EPOCHS,
    )

    final_history_path = EXPORT_DIR / "training_history_final_all_runs.json"
    with open(final_history_path, "w") as f:
        json.dump(final_history, f, indent=2)
    print(f"  Saved final training history: {final_history_path}")

    plot_single_training_curve(
        final_history,
        FIG_DIR / "training_curve_final_all_runs.png",
        "Final All-Runs Training Loss",
    )

    production_config = {
        "model": "FNIRSClassifier1D",
        "variant": "small_gn_16_32_64",
        "data_path": str(DATA_PATH),
        "fnirs_columns": FNIRS_COLUMNS,
        "n_channels": N_CHANNELS,
        "n_classes": N_CLASSES,
        "channel_types": "26 hbo + 26 hbr",
        "seq_len": seq_len,
        "target_seq_len": TARGET_SEQ_LEN,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "bandpass_description": BANDPASS_DESCRIPTION,
        "classifier_drop": CLASSIFIER_DROP,
        "conv_drop": CONV_DROP,
        "weight_decay": WEIGHT_DECAY,
        "lr": LR,
        "lr_schedule": "CosineAnnealingLR",
        "lr_schedule_t_max": NUM_EPOCHS,
        "lr_schedule_eta_min": LR * 0.01,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "label_smoothing": LABEL_SMOOTHING,
        "device": DEVICE,
        "random_seed": RANDOM_SEED,
        "n_trainable_parameters": count_parameters(final_model),
        "run_ids_used_for_final_training": run_ids,
        "class_mapping": {
            "0": "No-Robot",
            "1": "Robot",
        },
        "decision_threshold_robot": 0.5,
        "run_aggregation": "mean_window_probability",
        "preprocessing": {
            "sort_by": "time_index",
            "input_units": "HbO/HbR after bandpass and concentration conversion",
            "bandpass": BANDPASS_DESCRIPTION,
            "nan_handling": "nan/posinf/neginf replaced with 0.0 before normalization",
            "normalization": "per-channel z-score using final all-run training stats",
            "windowing": "one task_instance per window",
            "pad_truncate_target_seq_len": TARGET_SEQ_LEN,
            "input_shape": [N_CHANNELS, seq_len],
            "channel_order": "FNIRS_COLUMNS order in config",
        },
        "loocv_window_metrics": {
            k: round(float(v), 4) for k, v in aggregate_window.items()
        },
        "loocv_run_metrics": {
            k: round(float(v), 4) for k, v in aggregate_run.items()
        },
        "loocv_run_metrics_median_probability_diagnostic": {
            k: round(float(v), 4) for k, v in aggregate_run_median.items()
        },
        "loocv_run_metrics_majority_vote_diagnostic": {
            k: round(float(v), 4) for k, v in aggregate_run_majority.items()
        },
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
        target_seq_len=np.array([-1 if TARGET_SEQ_LEN is None else TARGET_SEQ_LEN]),
        sampling_rate_hz=np.array([SAMPLING_RATE_HZ]),
    )
    print(f"  Saved normalization stats: {norm_path}")

    print(f"\n{'=' * 70}")
    print("DONE — LOOCV evaluated; final all-runs production model exported")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
