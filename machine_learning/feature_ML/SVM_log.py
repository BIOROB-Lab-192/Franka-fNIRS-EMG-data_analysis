"""
EMG + fNIRS Handcrafted Feature Models — Binary Robot vs No-Robot
==================================================================

Models:
  1. Linear SVM
  2. L1 Logistic Regression

Validation:
  Outer leave-one-run-out CV for honest evaluation.
  Inner leave-one-run-out CV on training runs only for:
    - model hyperparameters
    - EMG/fNIRS feature-balance weights

Outputs:
  - aggregate metrics
  - per-window predictions
  - per-run predictions
  - grid search results
  - confusion matrices
  - ROC curves
  - final models trained on all runs
  - final scalers/configs

Main target:
  Binary classification: is_robot = 0/1
"""

import json
import csv
import joblib
import warnings
from pathlib import Path
from datetime import datetime

import polars as pl
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ──────────────────────────── CONFIG ────────────────────────────

EMG_PATH = Path("./data/processed/combined/data_packet/emg_rms.parquet")
FNIRS_PATH = Path("./data/processed/combined/data_packet/fnirs_full.parquet")

FIG_DIR = Path("./machine_learning/feature_ML/figures")
EXPORT_DIR = Path("./machine_learning/feature_ML/export")
MODEL_DIR = Path("./machine_learning/feature_ML/models")

FIG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RUN_COL = "run_id"
INSTANCE_COL = "task_instance"
LABEL_COL = "is_robot"

RANDOM_SEED = 42

# Model-selection target inside inner CV.
# Good options: "run_accuracy", "run_f1_robot", "run_auc", "window_accuracy"
SELECTION_METRIC = "run_accuracy"

# EMG/fNIRS feature balance grid.
# w_emg = 1.0, w_fnirs = 0.0 means EMG-only.
# w_emg = 0.0, w_fnirs = 1.0 means fNIRS-only.
# The combined models should usually use both, so you may later restrict this to 0.25–0.75.
WEIGHT_GRID = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
]

# Model hyperparameter grids.
SVM_C_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
LOGREG_C_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

# Main signal columns.
EMG_COLUMNS = [
    "Avanti Sensor 1 (82703) | EMG 1 (mV)",
    "Avanti Sensor 2 (82529) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 2 (mV)",
]

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


# ──────────────────────────── UTILS ────────────────────────────

def check_required_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def safe_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def auc_trapezoid(sig):
    try:
        return float(np.trapezoid(sig))
    except AttributeError:
        return float(np.trapz(sig))


def safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


# ──────────────────────────── FEATURE EXTRACTION ────────────────────────────

def channel_features(x, prefix, col_names):
    """
    x shape: T x C
    Returns one feature dictionary for a task window.
    """
    x = safe_np(x)
    feats = {}

    for i, name in enumerate(col_names):
        sig = x[:, i]
        base = f"{prefix}_{i}"

        feats[f"{base}_mean"] = float(np.mean(sig))
        feats[f"{base}_std"] = float(np.std(sig))
        feats[f"{base}_min"] = float(np.min(sig))
        feats[f"{base}_max"] = float(np.max(sig))
        feats[f"{base}_median"] = float(np.median(sig))
        feats[f"{base}_p25"] = float(np.percentile(sig, 25))
        feats[f"{base}_p75"] = float(np.percentile(sig, 75))
        feats[f"{base}_range"] = float(np.max(sig) - np.min(sig))
        feats[f"{base}_auc"] = auc_trapezoid(sig)

        n = len(sig)
        third = max(1, n // 3)

        early = float(np.mean(sig[:third]))
        middle = float(np.mean(sig[third:2 * third])) if n >= 3 else early
        late = float(np.mean(sig[-third:]))

        feats[f"{base}_early_mean"] = early
        feats[f"{base}_middle_mean"] = middle
        feats[f"{base}_late_mean"] = late
        feats[f"{base}_late_minus_early"] = late - early

        if n > 1:
            t = np.arange(n)
            slope = np.polyfit(t, sig, 1)[0]
        else:
            slope = 0.0

        feats[f"{base}_slope"] = float(slope)

    return feats


def extract_features_from_df(df, signal_cols, prefix, time_sort_col):
    rows = []

    run_ids = df[RUN_COL].unique().sort().to_list()

    for run_id in run_ids:
        run_df = df.filter(pl.col(RUN_COL) == run_id)
        label = int(run_df[LABEL_COL][0])

        instances = run_df[INSTANCE_COL].unique().sort().to_list()

        for inst in instances:
            inst_df = (
                run_df
                .filter(pl.col(INSTANCE_COL) == inst)
                .sort(time_sort_col)
            )

            x = inst_df.select(signal_cols).to_numpy()

            feats = {
                RUN_COL: run_id,
                INSTANCE_COL: int(inst),
                LABEL_COL: label,
            }

            feats.update(channel_features(x, prefix, signal_cols))
            rows.append(feats)

    return pd.DataFrame(rows)


def load_and_build_feature_table():
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    emg_df = pl.read_parquet(EMG_PATH)
    fnirs_df = pl.read_parquet(FNIRS_PATH)

    check_required_columns(
        emg_df,
        [RUN_COL, INSTANCE_COL, LABEL_COL, "sample_idx"] + EMG_COLUMNS,
        "EMG dataframe",
    )

    check_required_columns(
        fnirs_df,
        [RUN_COL, INSTANCE_COL, LABEL_COL, "time_index"] + FNIRS_COLUMNS,
        "fNIRS dataframe",
    )

    print(f"EMG rows:   {emg_df.shape[0]}, columns: {emg_df.shape[1]}")
    print(f"fNIRS rows: {fnirs_df.shape[0]}, columns: {fnirs_df.shape[1]}")

    print("\nExtracting EMG features...")
    emg_feat = extract_features_from_df(
        emg_df,
        EMG_COLUMNS,
        prefix="emg",
        time_sort_col="sample_idx",
    )

    print("Extracting fNIRS features...")
    fnirs_feat = extract_features_from_df(
        fnirs_df,
        FNIRS_COLUMNS,
        prefix="fnirs",
        time_sort_col="time_index",
    )

    print(f"EMG feature table:   {emg_feat.shape}")
    print(f"fNIRS feature table: {fnirs_feat.shape}")

    merged = emg_feat.merge(
        fnirs_feat,
        on=[RUN_COL, INSTANCE_COL, LABEL_COL],
        how="inner",
    )

    print(f"Combined feature table: {merged.shape}")

    run_labels = (
        merged[[RUN_COL, LABEL_COL]]
        .drop_duplicates()
        .sort_values(RUN_COL)
    )

    print("\nRuns:")
    for _, row in run_labels.iterrows():
        print(f"  {row[RUN_COL]}: label={int(row[LABEL_COL])}")

    emg_feature_cols = [c for c in merged.columns if c.startswith("emg_")]
    fnirs_feature_cols = [c for c in merged.columns if c.startswith("fnirs_")]

    print(f"\nFeature counts:")
    print(f"  EMG features:   {len(emg_feature_cols)}")
    print(f"  fNIRS features: {len(fnirs_feature_cols)}")
    print(f"  total features: {len(emg_feature_cols) + len(fnirs_feature_cols)}")

    merged.to_csv(EXPORT_DIR / "combined_handcrafted_features.csv", index=False)
    print(f"\nSaved feature table: {EXPORT_DIR / 'combined_handcrafted_features.csv'}")

    return merged, emg_feature_cols, fnirs_feature_cols


# ──────────────────────────── MODEL HELPERS ────────────────────────────

def make_model(model_name, C):
    if model_name == "linear_svm":
        return SVC(
            kernel="linear",
            C=C,
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        )

    if model_name == "logreg_l1":
        return LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_SEED,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def get_model_grid(model_name):
    if model_name == "linear_svm":
        return [{"C": c} for c in SVM_C_GRID]

    if model_name == "logreg_l1":
        return [{"C": c} for c in LOGREG_C_GRID]

    raise ValueError(f"Unknown model_name: {model_name}")


def build_weighted_features(df, feature_cols):
    return df[feature_cols].to_numpy(dtype=np.float64)


def fit_scaler_and_weight(X_train_raw, X_test_raw, emg_idx, fnirs_idx, w_emg):
    """
    Fit scaler on training data only, then apply modality weights.
    """
    w_fnirs = 1.0 - w_emg

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train[:, emg_idx] *= w_emg
    X_train[:, fnirs_idx] *= w_fnirs

    X_test[:, emg_idx] *= w_emg
    X_test[:, fnirs_idx] *= w_fnirs

    return X_train, X_test, scaler


def transform_with_existing_scaler(X_raw, scaler, emg_idx, fnirs_idx, w_emg):
    w_fnirs = 1.0 - w_emg
    X = scaler.transform(X_raw)
    X[:, emg_idx] *= w_emg
    X[:, fnirs_idx] *= w_fnirs
    return X


def predict_scores(model, X):
    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        prob = sigmoid(model.decision_function(X))
    else:
        prob = pred.astype(float)

    return pred.astype(int), prob.astype(float)


# ──────────────────────────── EVALUATION HELPERS ────────────────────────────

def compute_window_metrics(y_true, y_pred, y_prob):
    return {
        "window_accuracy": accuracy_score(y_true, y_pred),
        "window_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "window_precision_robot": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "window_recall_robot": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "window_f1_robot": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "window_precision_norobot": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "window_recall_norobot": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "window_f1_norobot": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "window_auc": safe_auc(y_true, y_prob),
    }


def aggregate_run_predictions(window_pred_df, threshold=0.5):
    rows = []

    for run_id, group in window_pred_df.groupby("held_out_run"):
        true_label = int(group["true_label"].iloc[0])
        prob_mean = float(group["prob_robot"].mean())
        prob_median = float(group["prob_robot"].median())
        pred_mean = int(prob_mean >= threshold)
        pred_median = int(prob_median >= threshold)

        rows.append({
            "held_out_run": run_id,
            "true_label": true_label,
            "prob_robot_mean": prob_mean,
            "prob_robot_median": prob_median,
            "pred_label_mean": pred_mean,
            "pred_label_median": pred_median,
            "correct_mean": pred_mean == true_label,
            "correct_median": pred_median == true_label,
            "n_windows": len(group),
            "window_accuracy": accuracy_score(group["true_label"], group["pred_label"]),
        })

    return pd.DataFrame(rows).sort_values("held_out_run")


def compute_run_metrics(run_df, pred_col="pred_label_mean", prob_col="prob_robot_mean"):
    y_true = run_df["true_label"].to_numpy(dtype=int)
    y_pred = run_df[pred_col].to_numpy(dtype=int)
    y_prob = run_df[prob_col].to_numpy(dtype=float)

    return {
        "run_accuracy": accuracy_score(y_true, y_pred),
        "run_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "run_precision_robot": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "run_recall_robot": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "run_f1_robot": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "run_precision_norobot": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "run_recall_norobot": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "run_f1_norobot": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "run_auc": safe_auc(y_true, y_prob),
    }


def threshold_search(run_df, prob_col="prob_robot_mean"):
    y_true = run_df["true_label"].to_numpy(dtype=int)
    y_prob = run_df[prob_col].to_numpy(dtype=float)

    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "threshold": float(threshold),
            "run_accuracy": accuracy_score(y_true, y_pred),
            "run_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "run_f1_robot": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "run_f1_norobot": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["run_accuracy", "run_balanced_accuracy", "run_f1_robot"],
        ascending=False,
    )
    return out


# ──────────────────────────── INNER GRID SEARCH ────────────────────────────

def evaluate_config_inner_cv(
    df_train_outer,
    feature_cols,
    emg_idx,
    fnirs_idx,
    model_name,
    C,
    w_emg,
):
    inner_runs = sorted(df_train_outer[RUN_COL].unique())

    all_window_rows = []

    for inner_held_out in inner_runs:
        inner_train = df_train_outer[df_train_outer[RUN_COL] != inner_held_out].copy()
        inner_val = df_train_outer[df_train_outer[RUN_COL] == inner_held_out].copy()

        X_train_raw = build_weighted_features(inner_train, feature_cols)
        y_train = inner_train[LABEL_COL].astype(int).to_numpy()

        X_val_raw = build_weighted_features(inner_val, feature_cols)
        y_val = inner_val[LABEL_COL].astype(int).to_numpy()

        X_train, X_val, _ = fit_scaler_and_weight(
            X_train_raw,
            X_val_raw,
            emg_idx,
            fnirs_idx,
            w_emg,
        )

        model = make_model(model_name, C=C)
        model.fit(X_train, y_train)

        pred, prob = predict_scores(model, X_val)

        for i in range(len(inner_val)):
            all_window_rows.append({
                "held_out_run": inner_held_out,
                "task_instance": int(inner_val[INSTANCE_COL].iloc[i]),
                "true_label": int(y_val[i]),
                "pred_label": int(pred[i]),
                "prob_robot": float(prob[i]),
            })

    window_df = pd.DataFrame(all_window_rows)
    run_df = aggregate_run_predictions(window_df, threshold=0.5)

    y_true_w = window_df["true_label"].to_numpy(dtype=int)
    y_pred_w = window_df["pred_label"].to_numpy(dtype=int)
    y_prob_w = window_df["prob_robot"].to_numpy(dtype=float)

    window_metrics = compute_window_metrics(y_true_w, y_pred_w, y_prob_w)
    run_metrics = compute_run_metrics(run_df)

    return {
        **window_metrics,
        **run_metrics,
    }


def inner_grid_search(
    df_train_outer,
    feature_cols,
    emg_idx,
    fnirs_idx,
    model_name,
):
    results = []

    for params in get_model_grid(model_name):
        C = params["C"]

        for w_emg in WEIGHT_GRID:
            metrics = evaluate_config_inner_cv(
                df_train_outer=df_train_outer,
                feature_cols=feature_cols,
                emg_idx=emg_idx,
                fnirs_idx=fnirs_idx,
                model_name=model_name,
                C=C,
                w_emg=w_emg,
            )

            row = {
                "model": model_name,
                "C": C,
                "w_emg": w_emg,
                "w_fnirs": 1.0 - w_emg,
                **metrics,
            }

            results.append(row)

    search_df = pd.DataFrame(results)

    # Selection with stable tie-breakers.
    sort_cols = [
        SELECTION_METRIC,
        "run_auc",
        "window_accuracy",
        "window_auc",
    ]

    search_df = search_df.sort_values(sort_cols, ascending=False)
    best = search_df.iloc[0].to_dict()

    return best, search_df


# ──────────────────────────── OUTER LOOCV ────────────────────────────

def run_nested_loocv(df, emg_feature_cols, fnirs_feature_cols, model_name):
    print("\n" + "=" * 80)
    print(f"OUTER LOOCV — {model_name}")
    print("=" * 80)

    feature_cols = emg_feature_cols + fnirs_feature_cols
    emg_idx = [feature_cols.index(c) for c in emg_feature_cols]
    fnirs_idx = [feature_cols.index(c) for c in fnirs_feature_cols]

    run_ids = sorted(df[RUN_COL].unique())

    outer_window_rows = []
    outer_grid_rows = []
    outer_selected_rows = []

    for fold_idx, held_out in enumerate(run_ids, start=1):
        print(f"\n--- Outer fold {fold_idx}/{len(run_ids)}: held out '{held_out}' ---")

        outer_train = df[df[RUN_COL] != held_out].copy()
        outer_test = df[df[RUN_COL] == held_out].copy()

        print("  Inner grid search...")
        best_config, inner_search_df = inner_grid_search(
            df_train_outer=outer_train,
            feature_cols=feature_cols,
            emg_idx=emg_idx,
            fnirs_idx=fnirs_idx,
            model_name=model_name,
        )

        print(
            f"  Best inner config: C={best_config['C']}, "
            f"w_emg={best_config['w_emg']:.2f}, "
            f"w_fnirs={best_config['w_fnirs']:.2f}, "
            f"inner_run_acc={best_config['run_accuracy']:.3f}, "
            f"inner_run_auc={best_config['run_auc']:.3f}"
        )

        inner_search_df["outer_fold"] = fold_idx
        inner_search_df["outer_held_out_run"] = held_out
        outer_grid_rows.extend(inner_search_df.to_dict("records"))

        outer_selected_rows.append({
            "outer_fold": fold_idx,
            "outer_held_out_run": held_out,
            "model": model_name,
            "selected_C": best_config["C"],
            "selected_w_emg": best_config["w_emg"],
            "selected_w_fnirs": best_config["w_fnirs"],
            "inner_run_accuracy": best_config["run_accuracy"],
            "inner_run_auc": best_config["run_auc"],
            "inner_window_accuracy": best_config["window_accuracy"],
            "inner_window_auc": best_config["window_auc"],
        })

        # Fit selected model on outer training data only.
        X_train_raw = build_weighted_features(outer_train, feature_cols)
        y_train = outer_train[LABEL_COL].astype(int).to_numpy()

        X_test_raw = build_weighted_features(outer_test, feature_cols)
        y_test = outer_test[LABEL_COL].astype(int).to_numpy()

        X_train, X_test, scaler = fit_scaler_and_weight(
            X_train_raw,
            X_test_raw,
            emg_idx,
            fnirs_idx,
            best_config["w_emg"],
        )

        model = make_model(model_name, C=best_config["C"])
        model.fit(X_train, y_train)

        pred, prob = predict_scores(model, X_test)

        fold_window_acc = accuracy_score(y_test, pred)
        print(f"  Outer test window acc: {fold_window_acc:.3f}")
        print(f"  Outer test mean prob_robot: {np.mean(prob):.3f}")

        for i in range(len(outer_test)):
            outer_window_rows.append({
                "model": model_name,
                "fold": fold_idx,
                "held_out_run": held_out,
                "task_instance": int(outer_test[INSTANCE_COL].iloc[i]),
                "true_label": int(y_test[i]),
                "pred_label": int(pred[i]),
                "prob_robot": float(prob[i]),
                "correct": int(pred[i]) == int(y_test[i]),
                "selected_C": best_config["C"],
                "selected_w_emg": best_config["w_emg"],
                "selected_w_fnirs": best_config["w_fnirs"],
            })

    window_df = pd.DataFrame(outer_window_rows)
    run_df = aggregate_run_predictions(window_df, threshold=0.5)
    run_df.insert(0, "model", model_name)

    grid_df = pd.DataFrame(outer_grid_rows)
    selected_df = pd.DataFrame(outer_selected_rows)

    y_true_w = window_df["true_label"].to_numpy(dtype=int)
    y_pred_w = window_df["pred_label"].to_numpy(dtype=int)
    y_prob_w = window_df["prob_robot"].to_numpy(dtype=float)

    window_metrics = compute_window_metrics(y_true_w, y_pred_w, y_prob_w)
    run_metrics = compute_run_metrics(run_df)

    aggregate = {
        "model": model_name,
        "evaluation": "nested leave-one-run-out CV",
        "selection_metric": SELECTION_METRIC,
        **window_metrics,
        **run_metrics,
    }

    print("\nAggregate:")
    for k, v in aggregate.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return aggregate, window_df, run_df, grid_df, selected_df


# ──────────────────────────── FINAL MODEL TRAINING ────────────────────────────

def choose_global_config_from_full_loocv(df, emg_feature_cols, fnirs_feature_cols, model_name):
    """
    Select final deployment config by LOOCV across all available runs.
    This is for training the final model on all data after evaluation.
    """
    feature_cols = emg_feature_cols + fnirs_feature_cols
    emg_idx = [feature_cols.index(c) for c in emg_feature_cols]
    fnirs_idx = [feature_cols.index(c) for c in fnirs_feature_cols]

    best_config, search_df = inner_grid_search(
        df_train_outer=df,
        feature_cols=feature_cols,
        emg_idx=emg_idx,
        fnirs_idx=fnirs_idx,
        model_name=model_name,
    )

    return best_config, search_df


def train_final_model(df, emg_feature_cols, fnirs_feature_cols, model_name, best_config):
    feature_cols = emg_feature_cols + fnirs_feature_cols
    emg_idx = [feature_cols.index(c) for c in emg_feature_cols]
    fnirs_idx = [feature_cols.index(c) for c in fnirs_feature_cols]

    X_raw = build_weighted_features(df, feature_cols)
    y = df[LABEL_COL].astype(int).to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    X[:, emg_idx] *= best_config["w_emg"]
    X[:, fnirs_idx] *= best_config["w_fnirs"]

    model = make_model(model_name, C=best_config["C"])
    model.fit(X, y)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "emg_feature_cols": emg_feature_cols,
        "fnirs_feature_cols": fnirs_feature_cols,
        "emg_idx": emg_idx,
        "fnirs_idx": fnirs_idx,
        "config": best_config,
        "label_mapping": {
            "0": "no_robot",
            "1": "robot",
        },
    }

    model_path = MODEL_DIR / f"final_{model_name}_emg_fnirs_features.joblib"
    joblib.dump(artifact, model_path)

    print(f"  Saved final model: {model_path}")

    return model_path


# ──────────────────────────── PLOTS ────────────────────────────

def plot_confusion(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

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

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5

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


def plot_roc_curve(y_true, y_prob, title, save_path):
    if len(np.unique(y_true)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
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

    return auc_val


def save_plots(model_name, window_df, run_df):
    # Window-level plots
    y_true_w = window_df["true_label"].to_numpy(dtype=int)
    y_pred_w = window_df["pred_label"].to_numpy(dtype=int)
    y_prob_w = window_df["prob_robot"].to_numpy(dtype=float)

    plot_confusion(
        y_true_w,
        y_pred_w,
        f"{model_name} — Window Confusion Matrix",
        FIG_DIR / f"{model_name}_window_confusion_matrix.png",
    )

    plot_roc_curve(
        y_true_w,
        y_prob_w,
        f"{model_name} — Window ROC",
        FIG_DIR / f"{model_name}_window_roc_curve.png",
    )

    # Run-level plots
    y_true_r = run_df["true_label"].to_numpy(dtype=int)
    y_pred_r = run_df["pred_label_mean"].to_numpy(dtype=int)
    y_prob_r = run_df["prob_robot_mean"].to_numpy(dtype=float)

    plot_confusion(
        y_true_r,
        y_pred_r,
        f"{model_name} — Run Confusion Matrix",
        FIG_DIR / f"{model_name}_run_confusion_matrix.png",
    )

    plot_roc_curve(
        y_true_r,
        y_prob_r,
        f"{model_name} — Run ROC",
        FIG_DIR / f"{model_name}_run_roc_curve.png",
    )


# ──────────────────────────── SAVE OUTPUTS ────────────────────────────

def save_model_outputs(
    model_name,
    aggregate,
    window_df,
    run_df,
    grid_df,
    selected_df,
    final_config,
    final_model_path,
):
    window_path = EXPORT_DIR / f"{model_name}_window_predictions.csv"
    run_path = EXPORT_DIR / f"{model_name}_run_predictions.csv"
    grid_path = EXPORT_DIR / f"{model_name}_nested_grid_results.csv"
    selected_path = EXPORT_DIR / f"{model_name}_selected_configs_by_fold.csv"
    metrics_path = EXPORT_DIR / f"{model_name}_metrics.csv"
    config_path = EXPORT_DIR / f"{model_name}_final_config.json"
    threshold_path = EXPORT_DIR / f"{model_name}_threshold_search.csv"

    window_df.to_csv(window_path, index=False)
    run_df.to_csv(run_path, index=False)
    grid_df.to_csv(grid_path, index=False)
    selected_df.to_csv(selected_path, index=False)

    threshold_df = threshold_search(run_df, prob_col="prob_robot_mean")
    threshold_df.to_csv(threshold_path, index=False)

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["date", datetime.now().isoformat()])
        writer.writerow(["model", model_name])
        writer.writerow(["evaluation", aggregate["evaluation"]])
        writer.writerow(["selection_metric", aggregate["selection_metric"]])
        writer.writerow(["n_outer_folds", run_df.shape[0]])
        writer.writerow(["n_windows", window_df.shape[0]])
        writer.writerow([])

        writer.writerow(["--- Nested LOOCV Aggregate ---", ""])
        for k, v in aggregate.items():
            if k not in ["model", "evaluation", "selection_metric"]:
                writer.writerow([k, v])

        writer.writerow([])
        writer.writerow(["--- Final Config Chosen From Full LOOCV ---", ""])
        for k, v in final_config.items():
            writer.writerow([k, v])

        writer.writerow([])
        writer.writerow(["--- Best Exploratory Threshold on Outer Predictions ---", ""])
        best_thresh = threshold_df.iloc[0].to_dict()
        for k, v in best_thresh.items():
            writer.writerow([k, v])

    config = {
        "model": model_name,
        "date": datetime.now().isoformat(),
        "data": {
            "emg_path": str(EMG_PATH),
            "fnirs_path": str(FNIRS_PATH),
            "run_col": RUN_COL,
            "instance_col": INSTANCE_COL,
            "label_col": LABEL_COL,
            "emg_columns": EMG_COLUMNS,
            "fnirs_columns": FNIRS_COLUMNS,
        },
        "evaluation": {
            "outer_cv": "leave-one-run-out",
            "inner_cv": "leave-one-run-out on training runs",
            "selection_metric": SELECTION_METRIC,
            "weight_grid": WEIGHT_GRID,
            "svm_c_grid": SVM_C_GRID,
            "logreg_c_grid": LOGREG_C_GRID,
        },
        "nested_loocv_aggregate": {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in aggregate.items()
        },
        "final_config": {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in final_config.items()
        },
        "final_model_path": str(final_model_path),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved outputs for {model_name}:")
    print(f"  {window_path}")
    print(f"  {run_path}")
    print(f"  {grid_path}")
    print(f"  {selected_path}")
    print(f"  {metrics_path}")
    print(f"  {config_path}")
    print(f"  {threshold_path}")


# ──────────────────────────── MAIN ────────────────────────────

def main():
    print("=" * 80)
    print("EMG + fNIRS HANDCRAFTED FEATURES — BINARY ROBOT VS NO-ROBOT")
    print("=" * 80)

    df, emg_feature_cols, fnirs_feature_cols = load_and_build_feature_table()

    model_names = ["linear_svm", "logreg_l1"]

    all_summary_rows = []

    for model_name in model_names:
        aggregate, window_df, run_df, grid_df, selected_df = run_nested_loocv(
            df=df,
            emg_feature_cols=emg_feature_cols,
            fnirs_feature_cols=fnirs_feature_cols,
            model_name=model_name,
        )

        print("\nChoosing final deployment config using full LOOCV grid...")
        final_config, full_search_df = choose_global_config_from_full_loocv(
            df=df,
            emg_feature_cols=emg_feature_cols,
            fnirs_feature_cols=fnirs_feature_cols,
            model_name=model_name,
        )

        full_search_path = EXPORT_DIR / f"{model_name}_full_loocv_grid_for_final_config.csv"
        full_search_df.to_csv(full_search_path, index=False)

        print(
            f"Final config for {model_name}: "
            f"C={final_config['C']}, "
            f"w_emg={final_config['w_emg']:.2f}, "
            f"w_fnirs={final_config['w_fnirs']:.2f}, "
            f"full_loocv_run_acc={final_config['run_accuracy']:.3f}, "
            f"full_loocv_run_auc={final_config['run_auc']:.3f}"
        )

        print("Training final model on all runs...")
        final_model_path = train_final_model(
            df=df,
            emg_feature_cols=emg_feature_cols,
            fnirs_feature_cols=fnirs_feature_cols,
            model_name=model_name,
            best_config=final_config,
        )

        save_plots(model_name, window_df, run_df)

        save_model_outputs(
            model_name=model_name,
            aggregate=aggregate,
            window_df=window_df,
            run_df=run_df,
            grid_df=grid_df,
            selected_df=selected_df,
            final_config=final_config,
            final_model_path=final_model_path,
        )

        all_summary_rows.append({
            "model": model_name,
            "nested_window_accuracy": aggregate["window_accuracy"],
            "nested_window_balanced_accuracy": aggregate["window_balanced_accuracy"],
            "nested_window_f1_robot": aggregate["window_f1_robot"],
            "nested_window_auc": aggregate["window_auc"],
            "nested_run_accuracy": aggregate["run_accuracy"],
            "nested_run_balanced_accuracy": aggregate["run_balanced_accuracy"],
            "nested_run_f1_robot": aggregate["run_f1_robot"],
            "nested_run_auc": aggregate["run_auc"],
            "final_C": final_config["C"],
            "final_w_emg": final_config["w_emg"],
            "final_w_fnirs": final_config["w_fnirs"],
            "final_full_loocv_run_accuracy": final_config["run_accuracy"],
            "final_full_loocv_run_auc": final_config["run_auc"],
            "final_model_path": str(final_model_path),
        })

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = EXPORT_DIR / "model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(
        summary_df
        .sort_values(["nested_run_accuracy", "nested_run_auc", "nested_window_accuracy"], ascending=False)
        .to_string(index=False)
    )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved figures to: {FIG_DIR}")
    print(f"Saved models to: {MODEL_DIR}")

    print("\nDONE ✓")


if __name__ == "__main__":
    main()