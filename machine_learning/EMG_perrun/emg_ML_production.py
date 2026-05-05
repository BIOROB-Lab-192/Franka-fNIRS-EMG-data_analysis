"""
EMG 1D CNN — Production Model
==============================
Winning config: 5036 timesteps, conv_dropout=0.2, classifier_dropout=0.3
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

N_CHANNELS = len(EMG_COLUMNS)
N_CLASSES = 2
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
RANDOM_SEED = 42
DEVICE = "cpu"

# Winning config from grid search
CLASSIFIER_DROP = 0.3
CONV_DROP = 0.2

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FIG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Config: conv_drop={CONV_DROP}, classifier_drop={CLASSIFIER_DROP}, "
      f"weight_decay={WEIGHT_DECAY}, lr={LR}")


# ──────────────────────────── DATA ────────────────────────────

def load_and_prepare():
    df = pl.read_parquet(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Runs: {df['run_id'].unique().to_list()}")
    print(f"Task instances: {df['task_instance'].n_unique()}")
    runs = {}
    for run_id in df["run_id"].unique().sort().to_list():
        run_df = df.filter(pl.col("run_id") == run_id)
        label = run_df["is_robot"][0]
        runs[run_id] = {"df": run_df, "label": int(label),
                         "n_instances": run_df["task_instance"].n_unique()}
        print(f"  {run_id}: label={label}, {runs[run_id]['n_instances']} instances")
    return df, runs


def compute_downsample_factor(runs, target_len=4200):
    first_run = list(runs.values())[0]
    inst = first_run["df"]["task_instance"].unique()[0]
    sample_df = first_run["df"].filter(pl.col("task_instance") == inst).sort("sample_idx")
    raw_len = sample_df.select(EMG_COLUMNS).to_numpy().shape[0]
    factor = max(1, raw_len // target_len)
    actual_len = raw_len // factor
    print(f"  Data: {raw_len} samples → downsample {factor}x → {actual_len} timesteps")
    return factor, actual_len


def preprocess_window(window_df, downsample_factor, target_len=None):
    raw = window_df.select(EMG_COLUMNS).to_numpy()
    ds = raw[::downsample_factor]
    if target_len is not None:
        if ds.shape[0] >= target_len:
            ds = ds[:target_len]
        else:
            pad = target_len - ds.shape[0]
            ds = np.pad(ds, ((0, pad), (0, 0)), mode="constant")
    ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0)
    return ds.T.astype(np.float32)


class EMGDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(np.array(windows), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def build_fold_data(runs, held_out_run, downsample_factor, target_len):
    train_windows, train_labels = [], []
    test_windows, test_labels = [], []
    for run_id, info in runs.items():
        run_df = info["df"]
        label = info["label"]
        instances = run_df["task_instance"].unique().sort().to_list()
        for inst in instances:
            inst_df = run_df.filter(pl.col("task_instance") == inst)
            window = inst_df.sort("sample_idx")
            if run_id == held_out_run:
                test_windows.append(preprocess_window(window, downsample_factor, target_len))
                test_labels.append(label)
            else:
                train_windows.append(preprocess_window(window, downsample_factor, target_len))
                train_labels.append(label)
    train_arr = np.array(train_windows)
    train_mean = train_arr.mean(axis=(0, 2), keepdims=True)
    train_std = train_arr.std(axis=(0, 2), keepdims=True) + 1e-8
    train_arr = (train_arr - train_mean) / train_std
    test_arr = (np.array(test_windows) - train_mean) / train_std
    print(f"  Train: {len(train_labels)} windows "
          f"(robot={sum(train_labels)}, norobot={len(train_labels)-sum(train_labels)})")
    print(f"  Test:  {len(test_labels)} windows "
          f"(robot={sum(test_labels)}, norobot={len(test_labels)-sum(test_labels)})")
    return EMGDataset(train_arr, train_labels), EMGDataset(test_arr, test_labels)


# ──────────────────────────── MODEL ────────────────────────────

class EMGClassifier1D(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, seq_len=5036, n_classes=N_CLASSES,
                 classifier_drop=CLASSIFIER_DROP, conv_drop=CONV_DROP):
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


# ──────────────────────────── TRAINING ────────────────────────────

def train_fold(model, train_loader, test_loader, fold_idx):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_test_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        test_loss /= len(test_loader.dataset)
        test_acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            print(f"    Epoch {epoch+1:3d} | train_loss={train_loss:.4f} | "
                  f"test_loss={test_loss:.4f} | test_acc={test_acc:.3f} "
                  f"{'*' if patience_counter == 0 else ''}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        model.cpu()

    # Final eval
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(dim=1).numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.numpy())

    return {
        "model": model, "history": history,
        "preds": np.array(all_preds), "labels": np.array(all_labels),
        "probs": np.array(all_probs),
    }


# ──────────────────────────── EVALUATION PLOTS ────────────────────────────

def plot_confusion_matrix(all_labels, all_preds, save_path):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No-Robot", "Robot"],
           yticklabels=["No-Robot", "Robot"], ylabel="True Label",
           xlabel="Predicted Label", title="Aggregate Confusion Matrix (All Folds)")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(all_labels, all_probs, save_path):
    robot_probs = all_probs[:, 1]
    valid = ~np.isnan(robot_probs)
    if valid.sum() < 2 or len(np.unique(all_labels[valid])) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(all_labels[valid], robot_probs[valid])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return roc_auc


def plot_fold_accuracy(fold_accs, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if a >= 0.5 else "#e74c3c" for a in fold_accs]
    ax.bar(range(1, len(fold_accs)+1), fold_accs, color=colors, edgecolor="white")
    ax.axhline(y=np.mean(fold_accs), color="gray", linestyle="--",
               label=f"Mean: {np.mean(fold_accs):.3f}")
    ax.set_xlabel("Fold (Held-out Run)"); ax.set_ylabel("Accuracy")
    ax.set_title("Per-Fold Accuracy (ds5036_drop_conv)"); ax.set_xticks(range(1, len(fold_accs)+1))
    ax.set_ylim([0, 1.05]); ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(all_histories, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    max_epochs = max(len(h["train_loss"]) for h in all_histories)
    train_losses = np.full((len(all_histories), max_epochs), np.nan)
    test_losses = np.full((len(all_histories), max_epochs), np.nan)
    test_accs = np.full((len(all_histories), max_epochs), np.nan)
    for i, h in enumerate(all_histories):
        n = len(h["train_loss"])
        train_losses[i, :n] = h["train_loss"]
        test_losses[i, :n] = h["test_loss"]
        test_accs[i, :n] = h["test_acc"]
    epochs = np.arange(1, max_epochs + 1)
    axes[0].plot(epochs, np.nanmean(train_losses, axis=0), label="Train", color="steelblue")
    axes[0].plot(epochs, np.nanmean(test_losses, axis=0), label="Test", color="coral")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Mean Loss Across Folds"); axes[0].legend()
    axes[1].plot(epochs, np.nanmean(test_accs, axis=0), color="forestgreen")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Mean Test Accuracy"); axes[1].set_ylim([0, 1.05])
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────── MAIN ────────────────────────────

def main():
    print("=" * 70)
    print("EMG 1D CNN — PRODUCTION MODEL (ds5036_drop_conv)")
    print("=" * 70)

    df, runs = load_and_prepare()
    run_ids = sorted(runs.keys())

    ds_factor, actual_seq_len = compute_downsample_factor(runs, target_len=4200)

    all_fold_metrics = []
    all_labels_agg, all_preds_agg, all_probs_agg = [], [], []
    all_histories = []
    fold_accuracies = []
    best_fold_acc = 0.0
    best_fold_model = None
    best_fold_idx = 0

    print(f"\n{'='*70}")
    print(f"Starting LOOCV: {len(run_ids)} folds")
    print(f"{'='*70}")

    for fold_idx, held_out in enumerate(run_ids):
        print(f"\n--- Fold {fold_idx+1}/{len(run_ids)}: held out '{held_out}' ---")
        train_ds, test_ds = build_fold_data(runs, held_out, ds_factor, actual_seq_len)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = EMGClassifier1D(n_channels=N_CHANNELS, seq_len=actual_seq_len)
        result = train_fold(model, train_loader, test_loader, fold_idx)

        acc = accuracy_score(result["labels"], result["preds"])
        prec = precision_score(result["labels"], result["preds"], pos_label=1, zero_division=0)
        rec = recall_score(result["labels"], result["preds"], pos_label=1, zero_division=0)
        f1 = f1_score(result["labels"], result["preds"], pos_label=1, zero_division=0)

        metrics = {
            "accuracy": acc, "precision_robot": prec,
            "recall_robot": rec, "f1_robot": f1,
            "precision_norobot": precision_score(result["labels"], result["preds"], pos_label=0, zero_division=0),
            "recall_norobot": recall_score(result["labels"], result["preds"], pos_label=0, zero_division=0),
            "f1_norobot": f1_score(result["labels"], result["preds"], pos_label=0, zero_division=0),
        }
        all_fold_metrics.append(metrics)
        fold_accuracies.append(acc)
        all_labels_agg.extend(result["labels"])
        all_preds_agg.extend(result["preds"])
        all_probs_agg.extend(result["probs"])
        all_histories.append(result["history"])

        print(f"  ✓ Accuracy: {acc:.3f} | F1_robot: {f1:.3f} | Recall_robot: {rec:.3f}")

        if acc > best_fold_acc:
            best_fold_acc = acc
            best_fold_model = result["model"]
            best_fold_idx = fold_idx + 1

    # Aggregate
    all_labels_agg = np.array(all_labels_agg)
    all_preds_agg = np.array(all_preds_agg)
    all_probs_agg = np.array(all_probs_agg)

    robot_probs = all_probs_agg[:, 1]
    valid = ~np.isnan(robot_probs)
    try:
        roc_auc = auc(*roc_curve(all_labels_agg[valid], robot_probs[valid])[:2])
    except Exception:
        roc_auc = 0.5

    aggregate = {
        "mean_accuracy": np.mean(fold_accuracies),
        "std_accuracy": np.std(fold_accuracies),
        "precision_robot": precision_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "recall_robot": recall_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "f1_robot": f1_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "precision_norobot": precision_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "recall_norobot": recall_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "f1_norobot": f1_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "roc_auc": roc_auc,
    }

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nPer-fold accuracies: {[f'{a:.3f}' for a in fold_accuracies]}")
    print(f"Best fold: #{best_fold_idx} (acc={best_fold_acc:.3f})")

    # ── Save plots ──
    print(f"\n{'='*70}")
    print("SAVING OUTPUTS")
    print(f"{'='*70}")
    plot_fold_accuracy(fold_accuracies, FIG_DIR / "prod_fold_accuracy.png")
    plot_confusion_matrix(all_labels_agg, all_preds_agg, FIG_DIR / "prod_confusion_matrix.png")
    plot_roc(all_labels_agg, all_probs_agg, FIG_DIR / "prod_roc_curve.png")
    plot_training_curves(all_histories, FIG_DIR / "prod_training_curves.png")
    print(f"  Saved 4 figures to {FIG_DIR}")

    # ── Save metrics CSV ──
    csv_path = EXPORT_DIR / "prod_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["date", datetime.now().isoformat()])
        writer.writerow(["model", "EMGClassifier1D (1D CNN) — ds5036_drop_conv"])
        writer.writerow(["config", "conv_drop=0.2, classifier_drop=0.3, weight_decay=1e-4"])
        writer.writerow(["downsample_factor", ds_factor])
        writer.writerow(["seq_len", actual_seq_len])
        writer.writerow(["n_folds", len(run_ids)])
        writer.writerow(["best_fold", best_fold_idx])
        writer.writerow([])
        writer.writerow(["--- Per-Fold ---", ""])
        for i, (m, rid) in enumerate(zip(all_fold_metrics, run_ids)):
            writer.writerow([f"fold_{i+1}_run", rid])
            for k, v in m.items():
                writer.writerow([f"fold_{i+1}_{k}", f"{v:.4f}"])
        writer.writerow([])
        writer.writerow(["--- Aggregate ---", ""])
        for k, v in aggregate.items():
            writer.writerow([k, f"{v:.4f}"])
    print(f"  Saved: {csv_path}")

    # ── Save best model ──
    model_path = EXPORT_DIR / "prod_model.pt"
    torch.save(best_fold_model.state_dict(), model_path)
    print(f"  Saved: {model_path}")

    # ── Save config ──
    config = {
        "model": "EMGClassifier1D",
        "variant": "ds5036_drop_conv",
        "data_path": str(DATA_PATH),
        "emg_columns": EMG_COLUMNS,
        "downsample_factor": ds_factor,
        "seq_len": actual_seq_len,
        "classifier_drop": CLASSIFIER_DROP,
        "conv_drop": CONV_DROP,
        "weight_decay": WEIGHT_DECAY,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "device": DEVICE,
        "run_ids": run_ids,
        "best_fold_idx": best_fold_idx,
        "best_fold_held_out": run_ids[best_fold_idx - 1],
        "aggregate_metrics": {k: round(v, 4) for k, v in aggregate.items()},
    }
    with open(EXPORT_DIR / "prod_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {EXPORT_DIR / 'prod_config.json'}")

    print(f"\n{'='*70}")
    print("DONE ✓ — Production model trained and exported")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
