"""
EMG 1D CNN Grid Search
======================
Tests 3 downsample rates × 4 regularization variants = 12 experiments.
Each experiment: 10-fold LOOCV by run_id.
"""

import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import csv
import json
from pathlib import Path
from datetime import datetime
import time
import sys

# ──────────────────────────── CONFIG ────────────────────────────

DATA_PATH = Path("./data/processed/combined/data_packet/emg_rms.parquet")
EXPORT_DIR = Path("./machine_learning/export")
FIG_DIR = Path("./machine_learning/figures")

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
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
RANDOM_SEED = 42
DEVICE = "cpu"

# Grid search space
DOWNSAMPLE_TARGETS = [4200, 2000, 500]   # target timesteps (coarse, medium, fine)
REGULARIZATION_VARIANTS = {
    "baseline":    {"classifier_drop": 0.3, "conv_drop": 0.0, "weight_decay": 1e-4},
    "drop_0.5":    {"classifier_drop": 0.5, "conv_drop": 0.0, "weight_decay": 1e-4},
    "drop_conv":   {"classifier_drop": 0.3, "conv_drop": 0.2, "weight_decay": 1e-4},
    "wd_1e3":      {"classifier_drop": 0.3, "conv_drop": 0.0, "weight_decay": 1e-3},
    "combined":    {"classifier_drop": 0.3, "conv_drop": 0.2, "weight_decay": 1e-3},
}

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────── DATA ────────────────────────────

def load_and_prepare():
    df = pl.read_parquet(DATA_PATH)
    runs = {}
    for run_id in df["run_id"].unique().sort().to_list():
        run_df = df.filter(pl.col("run_id") == run_id)
        label = run_df["is_robot"][0]
        runs[run_id] = {
            "df": run_df,
            "label": int(label),
            "n_instances": run_df["task_instance"].n_unique(),
        }
    return df, runs


def compute_downsample_factor(runs, target_len):
    first_run = list(runs.values())[0]
    inst = first_run["df"]["task_instance"].unique()[0]
    sample_df = first_run["df"].filter(pl.col("task_instance") == inst).sort("sample_idx")
    raw_len = sample_df.select(EMG_COLUMNS).to_numpy().shape[0]
    factor = max(1, raw_len // target_len)
    actual_len = raw_len // factor
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
    test_arr = np.array(test_windows)
    test_arr = (test_arr - train_mean) / train_std

    return (
        EMGDataset(train_arr, train_labels),
        EMGDataset(test_arr, test_labels),
    )


# ──────────────────────────── MODEL ────────────────────────────

class EMGClassifier1D(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, seq_len=2000, n_classes=N_CLASSES,
                 classifier_drop=0.3, conv_drop=0.0):
        super().__init__()
        self.use_conv_drop = conv_drop > 0

        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout1d(conv_drop) if conv_drop > 0 else nn.Identity(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(conv_drop) if conv_drop > 0 else nn.Identity(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(conv_drop) if conv_drop > 0 else nn.Identity(),
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


# ──────────────────────────── TRAINING ────────────────────────────

def train_fold(model, train_loader, test_loader, weight_decay):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_test_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    n_epochs_trained = 0

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

        scheduler.step(test_loss)
        n_epochs_trained = epoch + 1

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
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
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "n_epochs": n_epochs_trained,
    }


def compute_metrics(labels, preds, probs):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_robot": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_robot": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1_robot": f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision_norobot": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_norobot": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1_norobot": f1_score(labels, preds, pos_label=0, zero_division=0),
    }


# ──────────────────────────── GRID SEARCH ────────────────────────────

def run_one_experiment(runs, run_ids, ds_factor, actual_seq_len, reg_config, exp_name):
    """Run full LOOCV for one hyperparameter combination."""
    fold_accuracies = []
    fold_metrics_list = []
    all_labels_agg = []
    all_preds_agg = []
    all_probs_agg = []
    total_epochs = 0

    for fold_idx, held_out in enumerate(run_ids):
        train_ds, test_ds = build_fold_data(runs, held_out, ds_factor, actual_seq_len)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = EMGClassifier1D(
            n_channels=N_CHANNELS,
            seq_len=actual_seq_len,
            classifier_drop=reg_config["classifier_drop"],
            conv_drop=reg_config["conv_drop"],
        )

        result = train_fold(model, train_loader, test_loader, reg_config["weight_decay"])
        metrics = compute_metrics(result["labels"], result["preds"], result["probs"])

        fold_accuracies.append(metrics["accuracy"])
        fold_metrics_list.append(metrics)
        all_labels_agg.extend(result["labels"])
        all_preds_agg.extend(result["preds"])
        all_probs_agg.extend(result["probs"])
        total_epochs += result["n_epochs"]

    # Aggregate
    all_labels_agg = np.array(all_labels_agg)
    all_preds_agg = np.array(all_preds_agg)
    all_probs_agg = np.array(all_probs_agg)

    # ROC AUC
    robot_probs = all_probs_agg[:, 1]
    valid = ~np.isnan(robot_probs)
    if valid.sum() >= 2 and len(np.unique(all_labels_agg[valid])) >= 1:
        try:
            roc = roc_auc_score(all_labels_agg[valid], robot_probs[valid])
        except ValueError:
            roc = 0.5
    else:
        roc = 0.5

    return {
        "experiment": exp_name,
        "downsample_factor": ds_factor,
        "seq_len": actual_seq_len,
        "classifier_drop": reg_config["classifier_drop"],
        "conv_drop": reg_config["conv_drop"],
        "weight_decay": reg_config["weight_decay"],
        "mean_accuracy": np.mean(fold_accuracies),
        "std_accuracy": np.std(fold_accuracies),
        "min_accuracy": np.min(fold_accuracies),
        "max_accuracy": np.max(fold_accuracies),
        "precision_robot": precision_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "recall_robot": recall_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "f1_robot": f1_score(all_labels_agg, all_preds_agg, pos_label=1, zero_division=0),
        "precision_norobot": precision_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "recall_norobot": recall_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "f1_norobot": f1_score(all_labels_agg, all_preds_agg, pos_label=0, zero_division=0),
        "roc_auc": roc,
        "avg_epochs": total_epochs / len(run_ids),
        "fold_accuracies": fold_accuracies,
    }


def main():
    print("=" * 70)
    print("EMG 1D CNN — GRID SEARCH")
    print(f"  {len(DOWNSAMPLE_TARGETS)} downsample rates × {len(REGULARIZATION_VARIANTS)} reg variants = "
          f"{len(DOWNSAMPLE_TARGETS) * len(REGULARIZATION_VARIANTS)} experiments")
    print("=" * 70)

    # Load once
    df, runs = load_and_prepare()
    run_ids = sorted(runs.keys())
    print(f"Loaded {len(run_ids)} runs\n")

    all_results = []
    exp_num = 0
    total_exps = len(DOWNSAMPLE_TARGETS) * len(REGULARIZATION_VARIANTS)
    start_time = time.time()

    for target_len in DOWNSAMPLE_TARGETS:
        ds_factor, actual_seq_len = compute_downsample_factor(runs, target_len)
        print(f"\n{'─'*70}")
        print(f"Downsample target: {target_len} → factor={ds_factor}, actual_len={actual_seq_len}")
        print(f"{'─'*70}")

        for reg_name, reg_config in REGULARIZATION_VARIANTS.items():
            exp_num += 1
            exp_name = f"ds{actual_seq_len}_{reg_name}"
            print(f"\n[{exp_num}/{total_exps}] {exp_name}")
            print(f"  classifier_drop={reg_config['classifier_drop']}, "
                  f"conv_drop={reg_config['conv_drop']}, "
                  f"weight_decay={reg_config['weight_decay']}")

            exp_start = time.time()
            result = run_one_experiment(runs, run_ids, ds_factor, actual_seq_len, reg_config, exp_name)
            result["seq_len"] = actual_seq_len
            result["downsample_factor"] = ds_factor
            exp_time = time.time() - exp_start

            print(f"  → Accuracy: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f} "
                  f"(range {result['min_accuracy']:.3f}–{result['max_accuracy']:.3f}) "
                  f"| F1_robot: {result['f1_robot']:.3f} | AUC: {result['roc_auc']:.3f} "
                  f"| avg_epochs: {result['avg_epochs']:.0f} | {exp_time:.1f}s")

            all_results.append(result)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"GRID SEARCH COMPLETE — {total_time:.0f}s total")
    print(f"{'='*70}")

    # ── Save comparison CSV ──
    csv_path = EXPORT_DIR / "gridsearch_results.csv"
    csv_fields = [
        "experiment", "downsample_factor", "seq_len",
        "classifier_drop", "conv_drop", "weight_decay",
        "mean_accuracy", "std_accuracy", "min_accuracy", "max_accuracy",
        "precision_robot", "recall_robot", "f1_robot",
        "precision_norobot", "recall_norobot", "f1_norobot",
        "roc_auc", "avg_epochs",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\nSaved: {csv_path}")

    # ── Save best config as JSON ──
    best = max(all_results, key=lambda r: r["mean_accuracy"])
    best_config = {
        "experiment": best["experiment"],
        "downsample_factor": best["downsample_factor"],
        "seq_len": best["seq_len"],
        "classifier_drop": best["classifier_drop"],
        "conv_drop": best["conv_drop"],
        "weight_decay": best["weight_decay"],
        "mean_accuracy": best["mean_accuracy"],
        "std_accuracy": best["std_accuracy"],
        "roc_auc": best["roc_auc"],
    }
    with open(EXPORT_DIR / "gridsearch_best.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved: {EXPORT_DIR / 'gridsearch_best.json'}")

    # ── Print summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY (sorted by mean accuracy)")
    print(f"{'='*70}")
    print(f"{'Experiment':<30} {'Acc':>6} ± {'Std':>5}  {'F1_R':>5} {'F1_N':>5} {'AUC':>5}  {'SeqLen':>6}")
    print("-" * 80)
    sorted_results = sorted(all_results, key=lambda r: r["mean_accuracy"], reverse=True)
    for r in sorted_results:
        marker = " ★" if r["experiment"] == best["experiment"] else ""
        print(f"{r['experiment']:<30} {r['mean_accuracy']:>6.3f} ± {r['std_accuracy']:>5.3f}  "
              f"{r['f1_robot']:>5.3f} {r['f1_norobot']:>5.3f} {r['roc_auc']:>5.3f}  "
              f"{r['seq_len']:>6}{marker}")

    print(f"\nBest: {best['experiment']} → acc={best['mean_accuracy']:.3f} ± {best['std_accuracy']:.3f}")


if __name__ == "__main__":
    main()
