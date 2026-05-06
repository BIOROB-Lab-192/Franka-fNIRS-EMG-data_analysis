"""
Prior-Biased Confidence-Weighted Late Fusion — EMG + fNIRS
==========================================================

Fusion idea:
    confidence = abs(prob_robot - 0.5)

But confidence alone can overweight confidently-wrong models.

So this version uses:
    score_emg   = BASE_W_EMG   * confidence_emg
    score_fnirs = BASE_W_FNIRS * confidence_fnirs

Then:
    dynamic_w_emg   = score_emg / (score_emg + score_fnirs)
    dynamic_w_fnirs = score_fnirs / (score_emg + score_fnirs)

This keeps an EMG reliability prior while still allowing fNIRS to help.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

FNIRS_PATH = Path("./machine_learning/fNIRS_perrun/export/predictions_window_level_loocv.csv")
EMG_PATH = Path("./machine_learning/EMG_perrun/export/predictions_window_level_loocv.csv")

OUT_DIR = Path("./machine_learning/fusion_perrun/export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_OUT = OUT_DIR / "fusion_window_level_loocv_prior_confidence.csv"
RUN_OUT = OUT_DIR / "fusion_run_level_loocv_prior_confidence.csv"
SUMMARY_OUT = OUT_DIR / "fusion_summary_prior_confidence.csv"

# Prior reliability weights
BASE_W_EMG = 0.8
BASE_W_FNIRS = 0.2

THRESHOLD = 0.55
CONF_EPS = 1e-6

# Try 1.0 first. Higher values exaggerate confidence differences.
CONF_POWER = 1


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def get_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def print_metrics(title, y_true, y_pred, y_prob):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Accuracy:          {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision robot:   {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"Recall robot:      {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"F1 robot:          {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}")
    print(f"AUC:               {get_auc(y_true, y_prob):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


def metrics_dict(level, method, aggregation, y_true, y_pred, y_prob):
    return {
        "level": level,
        "method": method,
        "aggregation": aggregation,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_robot": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_robot": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_robot": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc": get_auc(y_true, y_prob),
    }


def standardize_prediction_df(df, modality):
    df = df.copy()

    if "prob_robot" in df.columns:
        prob_col = "prob_robot"
    elif "prob_robot_mean" in df.columns:
        prob_col = "prob_robot_mean"
    elif f"prob_robot_{modality}" in df.columns:
        prob_col = f"prob_robot_{modality}"
    else:
        raise ValueError(
            f"Could not find robot probability column in {modality} file. "
            f"Columns are: {list(df.columns)}"
        )

    if "true_label" in df.columns:
        label_col = "true_label"
    elif f"true_label_{modality}" in df.columns:
        label_col = f"true_label_{modality}"
    else:
        raise ValueError(
            f"Could not find true label column in {modality} file. "
            f"Columns are: {list(df.columns)}"
        )

    if "task_instance" not in df.columns:
        raise ValueError(
            f"{modality} file does not contain task_instance. "
            f"That means it is probably run-level, not window-level. "
            f"Use the window-level prediction CSV instead."
        )

    if "held_out_run" not in df.columns:
        raise ValueError(
            f"{modality} file does not contain held_out_run. "
            f"Columns are: {list(df.columns)}"
        )

    out = df[["held_out_run", "task_instance", label_col, prob_col]].rename(
        columns={
            label_col: f"true_label_{modality}",
            prob_col: f"prob_robot_{modality}",
        }
    )

    out["task_instance"] = out["task_instance"].astype(int)
    return out


# ---------------------------------------------------------------------
# LOAD WINDOW-LEVEL PREDICTIONS
# ---------------------------------------------------------------------

fnirs = pd.read_csv(FNIRS_PATH)
emg = pd.read_csv(EMG_PATH)

fnirs = standardize_prediction_df(fnirs, "fnirs")
emg = standardize_prediction_df(emg, "emg")

df = fnirs.merge(
    emg,
    on=["held_out_run", "task_instance"],
    how="inner",
)

if df.empty:
    raise ValueError("Merged dataframe is empty. Check held_out_run/task_instance alignment.")

if not np.all(df["true_label_fnirs"] == df["true_label_emg"]):
    bad = df[df["true_label_fnirs"] != df["true_label_emg"]]
    raise ValueError(f"Labels do not match between fNIRS and EMG files:\n{bad}")

df["true_label"] = df["true_label_fnirs"].astype(int)

print("\nLoaded and merged window-level predictions")
print("=" * 70)
print(f"Rows after merge: {len(df)}")
print(f"Runs: {df['held_out_run'].nunique()}")
print(f"Mean windows per run: {df.groupby('held_out_run').size().mean():.1f}")
print(f"Base weights: EMG={BASE_W_EMG}, fNIRS={BASE_W_FNIRS}")
print(f"Threshold: {THRESHOLD}")
print(f"Confidence power: {CONF_POWER}")


# ---------------------------------------------------------------------
# PRIOR-BIASED CONFIDENCE-WEIGHTED WINDOW FUSION
# ---------------------------------------------------------------------

df["conf_emg_raw"] = np.abs(df["prob_robot_emg"] - 0.5)
df["conf_fnirs_raw"] = np.abs(df["prob_robot_fnirs"] - 0.5)

df["conf_emg"] = np.power(df["conf_emg_raw"], CONF_POWER)
df["conf_fnirs"] = np.power(df["conf_fnirs_raw"], CONF_POWER)

df["score_emg"] = BASE_W_EMG * df["conf_emg"]
df["score_fnirs"] = BASE_W_FNIRS * df["conf_fnirs"]

df["score_total"] = df["score_emg"] + df["score_fnirs"] + CONF_EPS

df["dynamic_w_emg"] = df["score_emg"] / df["score_total"]
df["dynamic_w_fnirs"] = df["score_fnirs"] / df["score_total"]

df["prob_robot_fused_prior_conf"] = (
    df["dynamic_w_emg"] * df["prob_robot_emg"]
    + df["dynamic_w_fnirs"] * df["prob_robot_fnirs"]
)

df["pred_fused_prior_conf"] = (
    df["prob_robot_fused_prior_conf"] >= THRESHOLD
).astype(int)

df["correct_fused_prior_conf"] = (
    df["pred_fused_prior_conf"] == df["true_label"]
)


# ---------------------------------------------------------------------
# SAVE WINDOW-LEVEL OUTPUT
# ---------------------------------------------------------------------

window_cols = [
    "held_out_run",
    "task_instance",
    "true_label",
    "prob_robot_fnirs",
    "prob_robot_emg",

    "conf_fnirs_raw",
    "conf_emg_raw",
    "score_fnirs",
    "score_emg",
    "dynamic_w_fnirs",
    "dynamic_w_emg",

    "prob_robot_fused_prior_conf",
    "pred_fused_prior_conf",
    "correct_fused_prior_conf",
]

df[window_cols].to_csv(WINDOW_OUT, index=False)


# ---------------------------------------------------------------------
# WINDOW-LEVEL METRICS
# ---------------------------------------------------------------------

y_true_w = df["true_label"].values
y_pred_w = df["pred_fused_prior_conf"].values
y_prob_w = df["prob_robot_fused_prior_conf"].values

print_metrics(
    title=(
        f"WINDOW-LEVEL PRIOR-CONFIDENCE FUSION: "
        f"base EMG={BASE_W_EMG}, base fNIRS={BASE_W_FNIRS}, "
        f"threshold={THRESHOLD}, power={CONF_POWER}"
    ),
    y_true=y_true_w,
    y_pred=y_pred_w,
    y_prob=y_prob_w,
)

print("\nWindow-level preview:")
print(df[window_cols].head(30).to_string(index=False))


# ---------------------------------------------------------------------
# RUN-LEVEL AGGREGATION
# ---------------------------------------------------------------------

run_rows = []

for held_out_run, group in df.groupby("held_out_run"):
    true_label = int(group["true_label"].iloc[0])

    fused_mean = float(group["prob_robot_fused_prior_conf"].mean())
    fused_median = float(group["prob_robot_fused_prior_conf"].median())
    fused_vote_fraction = float(group["pred_fused_prior_conf"].mean())

    pred_mean = int(fused_mean >= THRESHOLD)
    pred_median = int(fused_median >= THRESHOLD)
    pred_majority = int(fused_vote_fraction >= 0.5)

    run_rows.append({
        "held_out_run": held_out_run,
        "true_label": true_label,
        "n_windows": int(len(group)),

        "prob_robot_fnirs_mean": float(group["prob_robot_fnirs"].mean()),
        "prob_robot_emg_mean": float(group["prob_robot_emg"].mean()),

        "dynamic_w_fnirs_mean": float(group["dynamic_w_fnirs"].mean()),
        "dynamic_w_emg_mean": float(group["dynamic_w_emg"].mean()),

        "prior_conf_fused_mean": fused_mean,
        "pred_prior_conf_mean": pred_mean,
        "correct_prior_conf_mean": pred_mean == true_label,

        "prior_conf_fused_median": fused_median,
        "pred_prior_conf_median": pred_median,
        "correct_prior_conf_median": pred_median == true_label,

        "prior_conf_window_vote_fraction": fused_vote_fraction,
        "pred_prior_conf_majority": pred_majority,
        "correct_prior_conf_majority": pred_majority == true_label,

        "window_accuracy_prior_conf_within_run": float(group["correct_fused_prior_conf"].mean()),
    })

run_df = pd.DataFrame(run_rows).sort_values("held_out_run")
run_df.to_csv(RUN_OUT, index=False)


# ---------------------------------------------------------------------
# RUN-LEVEL METRICS
# ---------------------------------------------------------------------

y_true_r = run_df["true_label"].values

print_metrics(
    title="RUN-LEVEL PRIOR-CONFIDENCE FUSION — MEAN PROBABILITY",
    y_true=y_true_r,
    y_pred=run_df["pred_prior_conf_mean"].values,
    y_prob=run_df["prior_conf_fused_mean"].values,
)

print_metrics(
    title="RUN-LEVEL PRIOR-CONFIDENCE FUSION — MEDIAN PROBABILITY",
    y_true=y_true_r,
    y_pred=run_df["pred_prior_conf_median"].values,
    y_prob=run_df["prior_conf_fused_median"].values,
)

print_metrics(
    title="RUN-LEVEL PRIOR-CONFIDENCE FUSION — MAJORITY VOTE",
    y_true=y_true_r,
    y_pred=run_df["pred_prior_conf_majority"].values,
    y_prob=run_df["prior_conf_window_vote_fraction"].values,
)


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

summary_rows = [
    metrics_dict(
        level="window",
        method="prior_confidence",
        aggregation="none",
        y_true=y_true_w,
        y_pred=df["pred_fused_prior_conf"].values,
        y_prob=df["prob_robot_fused_prior_conf"].values,
    ),
    metrics_dict(
        level="run",
        method="prior_confidence",
        aggregation="mean_probability",
        y_true=y_true_r,
        y_pred=run_df["pred_prior_conf_mean"].values,
        y_prob=run_df["prior_conf_fused_mean"].values,
    ),
    metrics_dict(
        level="run",
        method="prior_confidence",
        aggregation="median_probability",
        y_true=y_true_r,
        y_pred=run_df["pred_prior_conf_median"].values,
        y_prob=run_df["prior_conf_fused_median"].values,
    ),
    metrics_dict(
        level="run",
        method="prior_confidence",
        aggregation="majority_vote",
        y_true=y_true_r,
        y_pred=run_df["pred_prior_conf_majority"].values,
        y_prob=run_df["prior_conf_window_vote_fraction"].values,
    ),
]

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(
    summary_df.sort_values(
        ["level", "accuracy", "auc"],
        ascending=[True, False, False],
    ).to_string(index=False)
)


# ---------------------------------------------------------------------
# CORE ACCURACIES FINAL PRINT
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("CORE ACCURACIES")
print("=" * 70)

core = summary_df.copy()
core["accuracy_pct"] = (core["accuracy"] * 100).round(1)
core["auc_print"] = core["auc"].round(3)

for _, row in core.iterrows():
    print(
        f"{row['level']:>6} | "
        f"{row['method']:<18} | "
        f"{row['aggregation']:<18} | "
        f"acc={row['accuracy']:.4f} ({row['accuracy_pct']:.1f}%) | "
        f"bal_acc={row['balanced_accuracy']:.4f} | "
        f"f1_robot={row['f1_robot']:.4f} | "
        f"auc={row['auc_print']:.3f}"
    )

best_run = summary_df[summary_df["level"] == "run"].sort_values(
    ["accuracy", "auc"],
    ascending=False,
).iloc[0]

best_window = summary_df[summary_df["level"] == "window"].sort_values(
    ["accuracy", "auc"],
    ascending=False,
).iloc[0]

print("\nBEST WINDOW:")
print(
    f"{best_window['method']} / {best_window['aggregation']} "
    f"accuracy={best_window['accuracy']:.4f}, "
    f"balanced_accuracy={best_window['balanced_accuracy']:.4f}, "
    f"f1_robot={best_window['f1_robot']:.4f}, "
    f"auc={best_window['auc']:.4f}"
)

print("BEST RUN:")
print(
    f"{best_run['method']} / {best_run['aggregation']} "
    f"accuracy={best_run['accuracy']:.4f}, "
    f"balanced_accuracy={best_run['balanced_accuracy']:.4f}, "
    f"f1_robot={best_run['f1_robot']:.4f}, "
    f"auc={best_run['auc']:.4f}"
)

print("\nRun-level predictions:")
print(run_df.to_string(index=False))

print("\nSaved:")
print(f"  Window-level predictions: {WINDOW_OUT}")
print(f"  Run-level predictions:    {RUN_OUT}")
print(f"  Summary:                  {SUMMARY_OUT}")