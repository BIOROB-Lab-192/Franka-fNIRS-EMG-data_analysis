"""
EMG Robot vs No-Robot 1D CNN Classifier
========================================
Leave-One-Run-Out Cross-Validation using 4 EMG channels.
Classifies whether a trial was robot-assisted or no-robot.
"""

import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc,
    classification_report
)
import csv
import json
from pathlib import Path
from datetime import datetime

# ──────────────────────────── CONFIG ────────────────────────────

DATA_PATH = Path("./data/processed/combined/data_packet/emg_rms.parquet")
FIG_DIR = Path("./machine_learning/figures")
EXPORT_DIR = Path("./machine_learning/export")

EMG_COLUMNS = [
    "Avanti Sensor 1 (82703) | EMG 1 (mV)",
    "Avanti Sensor 2 (82529) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 2 (mV)",
]

DOWNSAMPLE_FACTOR = 13      # ~1259 Hz / 13 ≈ 97 Hz
TARGET_SEQ_LEN = 2000        # Will be recomputed from actual data length
N_CHANNELS = len(EMG_COLUMNS)
N_CLASSES = 2

BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
RANDOM_SEED = 42
DEVICE = "cpu"  # MPS Conv1d has NaN bugs; use CPU for stability

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FIG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")


# ──────────────────────────── DATA ────────────────────────────

def load_and_prepare():
    """Load parquet, group by run_id, return dict and metadata."""
    df = pl.read_parquet(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Runs: {df['run_id'].unique().to_list()}")
    print(f"Task instances: {df['task_instance'].n_unique()}")

    runs = {}
    for run_id in df["run_id"].unique().sort().to_list():
        run_df = df.filter(pl.col("run_id") == run_id)
        label = run_df["is_robot"][0]
        runs[run_id] = {
            "df": run_df,
            "label": int(label),
            "n_instances": run_df["task_instance"].n_unique(),
        }
        print(f"  {run_id}: label={label}, {runs[run_id]['n_instances']} instances")

    return df, runs


def preprocess_window(window_df, downsample_factor, target_len=None):
    """Extract EMG, downsample, z-score normalize, pad/truncate.
    If target_len is None, use full downsampled length.
    Returns (n_channels, T) float32 array.
    """
    raw = window_df.select(EMG_COLUMNS).to_numpy()  # (T, C)
    # Downsample
    ds = raw[::downsample_factor]  # (T//factor, C)
    if target_len is not None:
        if ds.shape[0] >= target_len:
            ds = ds[:target_len]
        else:
            pad = target_len - ds.shape[0]
            ds = np.pad(ds, ((0, pad), (0, 0)), mode="constant")
    # Replace any NaN/Inf with 0 (normalization done globally in build_fold_data)
    ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
    return ds.T.astype(np.float32)  # (C, T)


class EMGDataset(Dataset):
    """Dataset for preprocessed EMG windows."""

    def __init__(self, windows, labels):
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def compute_downsample_factor(runs, target_len=2000):
    """Compute downsample factor to reach target_len from actual data."""
    first_run = list(runs.values())[0]
    inst = first_run["df"]["task_instance"].unique()[0]
    sample_df = first_run["df"].filter(pl.col("task_instance") == inst).sort("sample_idx")
    raw_len = sample_df.select(EMG_COLUMNS).to_numpy().shape[0]
    factor = max(1, raw_len // target_len)
    actual_len = raw_len // factor
    print(f"  Data: {raw_len} samples → downsample {factor}x → {actual_len} timesteps")
    return factor, actual_len


def build_fold_data(runs, held_out_run, downsample_factor, target_len):
    """Build train/test datasets for one LOOCV fold.
    Normalization stats come from train only.
    """
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []

    for run_id, info in runs.items():
        run_df = info["df"]
        label = info["label"]
        instances = run_df["task_instance"].unique().sort().to_list()

        for inst in instances:
            inst_df = run_df.filter(pl.col("task_instance") == inst)
            # Group by time_sec to get unique timepoints
            window = inst_df.sort("sample_idx")

            if run_id == held_out_run:
                test_windows.append(preprocess_window(window, downsample_factor, target_len))
                test_labels.append(label)
            else:
                train_windows.append(preprocess_window(window, downsample_factor, target_len))
                train_labels.append(label)

    # Compute normalization stats from train (per channel)
    train_arr = np.array(train_windows)  # (N, C, T)
    train_mean = train_arr.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    train_std = train_arr.std(axis=(0, 2), keepdims=True) + 1e-8  # (1, C, 1)

    # Re-normalize train and test with train stats
    train_arr = (train_arr - train_mean) / train_std
    test_arr = np.array(test_windows)
    test_arr = (test_arr - train_mean) / train_std

    print(f"  Train: {len(train_labels)} windows "
          f"(robot={sum(train_labels)}, norobot={len(train_labels)-sum(train_labels)})")
    print(f"  Test:  {len(test_labels)} windows "
          f"(robot={sum(test_labels)}, norobot={len(test_labels)-sum(test_labels)})")

    return (
        EMGDataset(train_arr, train_labels),
        EMGDataset(test_arr, test_labels),
    )


# ──────────────────────────── MODEL ────────────────────────────

class EMGClassifier1D(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, seq_len=TARGET_SEQ_LEN, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
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
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_proba(self, x):
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# ──────────────────────────── TRAINING ────────────────────────────

def train_fold(model, train_loader, test_loader, device):
    """Train one fold. Returns dict with metrics and history."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_test_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Evaluate ---
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | "
                  f"test_loss={test_loss:.4f} | test_acc={test_acc:.3f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.cpu()

    # Final evaluation on test set
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.numpy())

    return {
        "model": model,
        "history": history,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
    }


# ──────────────────────────── EVALUATION ────────────────────────────

def compute_metrics(labels, preds, probs):
    """Compute all metrics for one fold."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_robot": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_robot": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_robot": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_norobot": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_norobot": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1_norobot": f1_score(labels, preds, pos_label=0, zero_division=0),
    }


def plot_confusion_matrix_aggregate(all_labels, all_preds, save_path):
    """Plot aggregate confusion matrix across all folds."""
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["No-Robot", "Robot"],
        yticklabels=["No-Robot", "Robot"],
        ylabel="True Label", xlabel="Predicted Label",
        title="Aggregate Confusion Matrix (All Folds)",
    )
    # Annotate
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_roc_curve(all_labels, all_probs, save_path):
    """Plot ROC curve across all folds."""
    # Use robot class (index 1) probabilities, filter NaN
    robot_probs = all_probs[:, 1]
    valid = ~np.isnan(robot_probs)
    if valid.sum() < 2 or len(np.unique(all_labels[valid])) < 2:
        print("  Skipping ROC curve: insufficient valid predictions or single class")
        # Save placeholder
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "ROC not available\n(single class predictions)", 
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("ROC Curve (Aggregate)")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return 0.5
    robot_probs = robot_probs[valid]
    labels_valid = all_labels[valid]
    fpr, tpr, _ = roc_curve(labels_valid, robot_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Aggregate)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return roc_auc


def plot_fold_accuracy(fold_accuracies, save_path):
    """Bar chart of per-fold accuracy."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if a >= 0.5 else "#e74c3c" for a in fold_accuracies]
    bars = ax.bar(range(1, len(fold_accuracies) + 1), fold_accuracies, color=colors, edgecolor="white")
    ax.axhline(y=np.mean(fold_accuracies), color="gray", linestyle="--", label=f"Mean: {np.mean(fold_accuracies):.3f}")
    ax.set_xlabel("Fold (Held-out Run)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Fold Accuracy (Leave-One-Run-Out)")
    ax.set_xticks(range(1, len(fold_accuracies) + 1))
    ax.set_ylim([0, 1.05])
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(all_histories, save_path):
    """Plot average training loss across folds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Train loss
    max_epochs = max(len(h["train_loss"]) for h in all_histories)
    train_losses = np.full((len(all_histories), max_epochs), np.nan)
    test_losses = np.full((len(all_histories), max_epochs), np.nan)
    for i, h in enumerate(all_histories):
        n = len(h["train_loss"])
        train_losses[i, :n] = h["train_loss"]
        test_losses[i, :n] = h["test_loss"]

    epochs = np.arange(1, max_epochs + 1)
    mean_train = np.nanmean(train_losses, axis=0)
    mean_test = np.nanmean(test_losses, axis=0)

    axes[0].plot(epochs, mean_train, label="Train Loss", color="steelblue")
    axes[0].plot(epochs, mean_test, label="Test Loss", color="coral")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Mean Training Loss Across Folds")
    axes[0].legend()

    # Test accuracy
    test_accs = np.full((len(all_histories), max_epochs), np.nan)
    for i, h in enumerate(all_histories):
        n = len(h["test_acc"])
        test_accs[i, :n] = h["test_acc"]
    mean_acc = np.nanmean(test_accs, axis=0)

    axes[1].plot(epochs, mean_acc, color="forestgreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Mean Test Accuracy Across Folds")
    axes[1].set_ylim([0, 1.05])
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────── SAVE EXPORTS ────────────────────────────

def save_metrics_csv(fold_metrics, aggregate, run_ids, save_path):
    """Save per-fold and aggregate metrics to CSV."""
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["date", datetime.now().isoformat()])
        writer.writerow(["model", "EMGClassifier1D (1D CNN)"])
        writer.writerow(["n_folds", len(fold_metrics)])
        writer.writerow(["downsample_factor", DOWNSAMPLE_FACTOR])
        writer.writerow(["target_seq_len", TARGET_SEQ_LEN])
        writer.writerow(["n_channels", N_CHANNELS])
        writer.writerow(["device", DEVICE])
        writer.writerow([])
        writer.writerow(["--- Per-Fold Results ---", ""])
        for i, (metrics, run_id) in enumerate(zip(fold_metrics, run_ids)):
            writer.writerow([f"fold_{i+1}_held_out_run", run_id])
            for k, v in metrics.items():
                writer.writerow([f"fold_{i+1}_{k}", f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Aggregate ---", ""])
        for k, v in aggregate.items():
            writer.writerow([k, f"{v:.4f}"])
    print(f"  Saved: {save_path}")


def save_best_model(model, save_path):
    """Save best model weights."""
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}")


# ──────────────────────────── MAIN ────────────────────────────

def main():
    print("=" * 60)
    print("EMG Robot vs No-Robot — 1D CNN LOOCV")
    print("=" * 60)

    # Load data
    df, runs = load_and_prepare()
    run_ids = sorted(runs.keys())

    # Auto-compute downsample factor for ~2000 timesteps
    ds_factor, actual_seq_len = compute_downsample_factor(runs, target_len=2000)

    all_fold_metrics = []
    all_labels_agg = []
    all_preds_agg = []
    all_probs_agg = []
    all_histories = []
    fold_accuracies = []
    best_fold_acc = 0.0
    best_fold_model = None

    print(f"\n{'='*60}")
    print(f"Starting LOOCV: {len(run_ids)} folds")
    print(f"{'='*60}")

    for fold_idx, held_out in enumerate(run_ids):
        print(f"\n--- Fold {fold_idx+1}/{len(run_ids)}: held out '{held_out}' ---")

        # Build data
        train_ds, test_ds = build_fold_data(runs, held_out, ds_factor, actual_seq_len)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Train
        model = EMGClassifier1D(n_channels=N_CHANNELS, seq_len=actual_seq_len)
        result = train_fold(model, train_loader, test_loader, DEVICE)

        # Metrics
        metrics = compute_metrics(result["labels"], result["preds"], result["probs"])
        all_fold_metrics.append(metrics)
        fold_accuracies.append(metrics["accuracy"])
        all_labels_agg.extend(result["labels"])
        all_preds_agg.extend(result["preds"])
        all_probs_agg.extend(result["probs"])
        all_histories.append(result["history"])

        print(f"  Accuracy: {metrics['accuracy']:.3f}")

        # Track best model
        if metrics["accuracy"] > best_fold_acc:
            best_fold_acc = metrics["accuracy"]
            best_fold_model = result["model"]

    # ── Aggregate metrics ──
    all_labels_agg = np.array(all_labels_agg)
    all_preds_agg = np.array(all_preds_agg)
    all_probs_agg = np.array(all_probs_agg)

    aggregate = {
        "mean_accuracy": np.mean(fold_accuracies),
        "std_accuracy": np.std(fold_accuracies),
        "precision_robot": precision_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "recall_robot": recall_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "f1_robot": f1_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "precision_norobot": precision_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "recall_norobot": recall_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "f1_norobot": f1_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
    }

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nPer-fold accuracies: {[f'{a:.3f}' for a in fold_accuracies]}")

    # ── Save everything ──
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print(f"{'='*60}")

    plot_fold_accuracy(fold_accuracies, FIG_DIR / "fold_accuracy.png")
    plot_confusion_matrix_aggregate(all_labels_agg, all_preds_agg, FIG_DIR / "confusion_matrix.png")
    roc_auc = plot_roc_curve(all_labels_agg, all_probs_agg, FIG_DIR / "roc_curve.png")
    plot_training_curves(all_histories, FIG_DIR / "training_curves.png")
    aggregate["roc_auc"] = roc_auc

    save_metrics_csv(all_fold_metrics, aggregate, run_ids, EXPORT_DIR / "metrics.csv")
    save_best_model(best_fold_model, EXPORT_DIR / "best_model.pt")

    # Save config as JSON for reproducibility
    config = {
        "data_path": str(DATA_PATH),
        "emg_columns": EMG_COLUMNS,
        "downsample_factor": ds_factor,
        "actual_seq_len": actual_seq_len,
        "n_channels": N_CHANNELS,
        "n_classes": N_CLASSES,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "device": DEVICE,
        "run_ids": run_ids,
        "best_fold_held_out": run_ids[np.argmax(fold_accuracies)],
    }
    with open(EXPORT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {EXPORT_DIR / 'config.json'}")

    print(f"\n{'='*60}")
    print("DONE ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
