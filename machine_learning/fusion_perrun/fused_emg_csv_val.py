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

WINDOW_OUT = OUT_DIR / "fusion_window_level_loocv_compare_methods.csv"
RUN_OUT = OUT_DIR / "fusion_run_level_loocv_compare_methods.csv"

# Pre-specified fusion rule
W_FNIRS = 0.2
W_EMG = 0.8
THRESHOLD = 0.55

EPS = 1e-6


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


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def standardize_prediction_df(df, modality):
    """
    Handles likely column names from previous scripts.

    Required output:
      held_out_run
      task_instance
      true_label_<modality>
      prob_robot_<modality>
    """

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

    keep = ["held_out_run", "task_instance", label_col, prob_col]

    out = df[keep].rename(
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
print(f"Fusion weights: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}")


# ---------------------------------------------------------------------
# WINDOW-LEVEL FUSION METHOD 1: PROBABILITY AVERAGE
# ---------------------------------------------------------------------

df["prob_robot_fused_probavg"] = (
    W_FNIRS * df["prob_robot_fnirs"]
    + W_EMG * df["prob_robot_emg"]
)

df["pred_fused_probavg"] = (
    df["prob_robot_fused_probavg"] >= THRESHOLD
).astype(int)

df["correct_fused_probavg"] = (
    df["pred_fused_probavg"] == df["true_label"]
)


# ---------------------------------------------------------------------
# WINDOW-LEVEL FUSION METHOD 2: LOGIT FUSION
# ---------------------------------------------------------------------

df["logit_robot_fnirs"] = logit(df["prob_robot_fnirs"])
df["logit_robot_emg"] = logit(df["prob_robot_emg"])

df["logit_robot_fused"] = (
    W_FNIRS * df["logit_robot_fnirs"]
    + W_EMG * df["logit_robot_emg"]
)

df["prob_robot_fused_logit"] = sigmoid(df["logit_robot_fused"])

df["pred_fused_logit"] = (
    df["prob_robot_fused_logit"] >= THRESHOLD
).astype(int)

df["correct_fused_logit"] = (
    df["pred_fused_logit"] == df["true_label"]
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

    "prob_robot_fused_probavg",
    "pred_fused_probavg",
    "correct_fused_probavg",

    "prob_robot_fused_logit",
    "pred_fused_logit",
    "correct_fused_logit",
]

df[window_cols].to_csv(WINDOW_OUT, index=False)


# ---------------------------------------------------------------------
# PRINT WINDOW-LEVEL METRICS
# ---------------------------------------------------------------------

y_true_w = df["true_label"].values

print_metrics(
    title=f"WINDOW-LEVEL PROBABILITY FUSION: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_w,
    y_pred=df["pred_fused_probavg"].values,
    y_prob=df["prob_robot_fused_probavg"].values,
)

print_metrics(
    title=f"WINDOW-LEVEL LOGIT FUSION: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_w,
    y_pred=df["pred_fused_logit"].values,
    y_prob=df["prob_robot_fused_logit"].values,
)

print("\nWindow-level preview:")
print(df[window_cols].head(30).to_string(index=False))


# ---------------------------------------------------------------------
# RUN-LEVEL AGGREGATION FROM WINDOW-LEVEL FUSION
# ---------------------------------------------------------------------

run_rows = []

for held_out_run, group in df.groupby("held_out_run"):
    true_label = int(group["true_label"].iloc[0])

    prob_robot_fnirs_mean = float(group["prob_robot_fnirs"].mean())
    prob_robot_emg_mean = float(group["prob_robot_emg"].mean())

    # Probability-average fusion aggregated by mean probability
    probavg_mean = float(group["prob_robot_fused_probavg"].mean())
    probavg_median = float(group["prob_robot_fused_probavg"].median())
    probavg_majority_vote = float(group["pred_fused_probavg"].mean())

    pred_probavg_mean = int(probavg_mean >= THRESHOLD)
    pred_probavg_median = int(probavg_median >= THRESHOLD)
    pred_probavg_majority = int(probavg_majority_vote >= 0.5)

    # Logit fusion aggregated by mean probability
    logit_mean = float(group["prob_robot_fused_logit"].mean())
    logit_median = float(group["prob_robot_fused_logit"].median())
    logit_majority_vote = float(group["pred_fused_logit"].mean())

    pred_logit_mean = int(logit_mean >= THRESHOLD)
    pred_logit_median = int(logit_median >= THRESHOLD)
    pred_logit_majority = int(logit_majority_vote >= 0.5)

    run_rows.append({
        "held_out_run": held_out_run,
        "true_label": true_label,
        "n_windows": int(len(group)),

        "prob_robot_fnirs_mean": prob_robot_fnirs_mean,
        "prob_robot_emg_mean": prob_robot_emg_mean,

        "probavg_mean": probavg_mean,
        "pred_probavg_mean": pred_probavg_mean,
        "correct_probavg_mean": pred_probavg_mean == true_label,

        "probavg_median": probavg_median,
        "pred_probavg_median": pred_probavg_median,
        "correct_probavg_median": pred_probavg_median == true_label,

        "probavg_window_vote_fraction": probavg_majority_vote,
        "pred_probavg_majority": pred_probavg_majority,
        "correct_probavg_majority": pred_probavg_majority == true_label,

        "logit_mean": logit_mean,
        "pred_logit_mean": pred_logit_mean,
        "correct_logit_mean": pred_logit_mean == true_label,

        "logit_median": logit_median,
        "pred_logit_median": pred_logit_median,
        "correct_logit_median": pred_logit_median == true_label,

        "logit_window_vote_fraction": logit_majority_vote,
        "pred_logit_majority": pred_logit_majority,
        "correct_logit_majority": pred_logit_majority == true_label,

        "window_accuracy_probavg_within_run": float(group["correct_fused_probavg"].mean()),
        "window_accuracy_logit_within_run": float(group["correct_fused_logit"].mean()),
    })

run_df = pd.DataFrame(run_rows).sort_values("held_out_run")
run_df.to_csv(RUN_OUT, index=False)


# ---------------------------------------------------------------------
# PRINT RUN-LEVEL METRICS
# ---------------------------------------------------------------------

y_true_r = run_df["true_label"].values

print_metrics(
    title=f"RUN-LEVEL PROBABILITY FUSION — MEAN PROBABILITY: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_r,
    y_pred=run_df["pred_probavg_mean"].values,
    y_prob=run_df["probavg_mean"].values,
)

print_metrics(
    title=f"RUN-LEVEL PROBABILITY FUSION — MEDIAN PROBABILITY: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_r,
    y_pred=run_df["pred_probavg_median"].values,
    y_prob=run_df["probavg_median"].values,
)

print_metrics(
    title=f"RUN-LEVEL PROBABILITY FUSION — MAJORITY VOTE: EMG={W_EMG}, fNIRS={W_FNIRS}",
    y_true=y_true_r,
    y_pred=run_df["pred_probavg_majority"].values,
    y_prob=run_df["probavg_window_vote_fraction"].values,
)

print_metrics(
    title=f"RUN-LEVEL LOGIT FUSION — MEAN PROBABILITY: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_r,
    y_pred=run_df["pred_logit_mean"].values,
    y_prob=run_df["logit_mean"].values,
)

print_metrics(
    title=f"RUN-LEVEL LOGIT FUSION — MEDIAN PROBABILITY: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_r,
    y_pred=run_df["pred_logit_median"].values,
    y_prob=run_df["logit_median"].values,
)

print_metrics(
    title=f"RUN-LEVEL LOGIT FUSION — MAJORITY VOTE: EMG={W_EMG}, fNIRS={W_FNIRS}",
    y_true=y_true_r,
    y_pred=run_df["pred_logit_majority"].values,
    y_prob=run_df["logit_window_vote_fraction"].values,
)


# ---------------------------------------------------------------------
# COMPACT SUMMARY
# ---------------------------------------------------------------------

summary_rows = [
    {
        "level": "window",
        "method": "probavg",
        "aggregation": "none",
        "accuracy": accuracy_score(y_true_w, df["pred_fused_probavg"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_w, df["pred_fused_probavg"]),
        "f1_robot": f1_score(y_true_w, df["pred_fused_probavg"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_w, df["prob_robot_fused_probavg"]),
    },
    {
        "level": "window",
        "method": "logit",
        "aggregation": "none",
        "accuracy": accuracy_score(y_true_w, df["pred_fused_logit"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_w, df["pred_fused_logit"]),
        "f1_robot": f1_score(y_true_w, df["pred_fused_logit"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_w, df["prob_robot_fused_logit"]),
    },
    {
        "level": "run",
        "method": "probavg",
        "aggregation": "mean_probability",
        "accuracy": accuracy_score(y_true_r, run_df["pred_probavg_mean"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_r, run_df["pred_probavg_mean"]),
        "f1_robot": f1_score(y_true_r, run_df["pred_probavg_mean"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_r, run_df["probavg_mean"]),
    },
    {
        "level": "run",
        "method": "probavg",
        "aggregation": "median_probability",
        "accuracy": accuracy_score(y_true_r, run_df["pred_probavg_median"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_r, run_df["pred_probavg_median"]),
        "f1_robot": f1_score(y_true_r, run_df["pred_probavg_median"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_r, run_df["probavg_median"]),
    },
    {
        "level": "run",
        "method": "probavg",
        "aggregation": "majority_vote",
        "accuracy": accuracy_score(y_true_r, run_df["pred_probavg_majority"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_r, run_df["pred_probavg_majority"]),
        "f1_robot": f1_score(y_true_r, run_df["pred_probavg_majority"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_r, run_df["probavg_window_vote_fraction"]),
    },
    {
        "level": "run",
        "method": "logit",
        "aggregation": "mean_probability",
        "accuracy": accuracy_score(y_true_r, run_df["pred_logit_mean"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_r, run_df["pred_logit_mean"]),
        "f1_robot": f1_score(y_true_r, run_df["pred_logit_mean"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_r, run_df["logit_mean"]),
    },
    {
        "level": "run",
        "method": "logit",
        "aggregation": "median_probability",
        "accuracy": accuracy_score(y_true_r, run_df["pred_logit_median"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_r, run_df["pred_logit_median"]),
        "f1_robot": f1_score(y_true_r, run_df["pred_logit_median"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_r, run_df["logit_median"]),
    },
    {
        "level": "run",
        "method": "logit",
        "aggregation": "majority_vote",
        "accuracy": accuracy_score(y_true_r, run_df["pred_logit_majority"]),
        "balanced_accuracy": balanced_accuracy_score(y_true_r, run_df["pred_logit_majority"]),
        "f1_robot": f1_score(y_true_r, run_df["pred_logit_majority"], pos_label=1, zero_division=0),
        "auc": get_auc(y_true_r, run_df["logit_window_vote_fraction"]),
    },
]

summary_df = pd.DataFrame(summary_rows)
summary_out = OUT_DIR / "fusion_summary_compare_methods.csv"
summary_df.to_csv(summary_out, index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(summary_df.sort_values(["level", "accuracy"], ascending=[True, False]).to_string(index=False))

print("\nRun-level predictions:")
print(run_df.to_string(index=False))

print("\nSaved:")
print(f"  Window-level predictions: {WINDOW_OUT}")
print(f"  Run-level predictions:    {RUN_OUT}")
print(f"  Summary:                  {summary_out}")

print("\n" + "=" * 70)
print("CORE ACCURACIES")
print("=" * 70)

core = summary_df.copy()
core["accuracy_pct"] = (core["accuracy"] * 100).round(1)
core["auc"] = core["auc"].round(3)

for _, row in core.iterrows():
    print(
        f"{row['level']:>6} | "
        f"{row['method']:<7} | "
        f"{row['aggregation']:<18} | "
        f"acc={row['accuracy']:.4f} ({row['accuracy_pct']:.1f}%) | "
        f"bal_acc={row['balanced_accuracy']:.4f} | "
        f"f1_robot={row['f1_robot']:.4f} | "
        f"auc={row['auc']:.3f}"
    )

best_run = summary_df[summary_df["level"] == "run"].sort_values(
    "accuracy", ascending=False
).iloc[0]

best_window = summary_df[summary_df["level"] == "window"].sort_values(
    "accuracy", ascending=False
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