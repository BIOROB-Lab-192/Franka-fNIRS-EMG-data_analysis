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

WINDOW_OUT = OUT_DIR / "fusion_window_level_loocv_weighted.csv"
RUN_OUT = OUT_DIR / "fusion_run_level_loocv_weighted.csv"

# Pre-specified fusion rule
W_FNIRS = 0.2
W_EMG = 0.8
THRESHOLD = 0.55


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


def standardize_prediction_df(df, modality):
    """
    Tries to handle a few possible column names from your earlier scripts.
    Required output:
      held_out_run, task_instance, true_label, prob_robot_<modality>
    """

    df = df.copy()

    # Handle probability column names
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

    # Handle true label column names
    if "true_label" in df.columns:
        label_col = "true_label"
    elif f"true_label_{modality}" in df.columns:
        label_col = f"true_label_{modality}"
    else:
        raise ValueError(
            f"Could not find true label column in {modality} file. "
            f"Columns are: {list(df.columns)}"
        )

    # Need task_instance for window-level fusion
    if "task_instance" not in df.columns:
        raise ValueError(
            f"{modality} file does not contain task_instance. "
            f"That means it is probably run-level, not window-level. "
            f"Use the window-level prediction CSV instead."
        )

    keep = ["held_out_run", "task_instance", label_col, prob_col]
    out = df[keep].rename(
        columns={
            label_col: f"true_label_{modality}",
            prob_col: f"prob_robot_{modality}",
        }
    )

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

# ---------------------------------------------------------------------
# WINDOW-LEVEL FUSION
# ---------------------------------------------------------------------

df["prob_robot_fused"] = (
    W_FNIRS * df["prob_robot_fnirs"]
    + W_EMG * df["prob_robot_emg"]
)

df["pred_fused"] = (df["prob_robot_fused"] >= THRESHOLD).astype(int)
df["correct_fused"] = df["pred_fused"] == df["true_label"]

window_cols = [
    "held_out_run",
    "task_instance",
    "true_label",
    "prob_robot_fnirs",
    "prob_robot_emg",
    "prob_robot_fused",
    "pred_fused",
    "correct_fused",
]

df[window_cols].to_csv(WINDOW_OUT, index=False)

y_true_w = df["true_label"].values
y_pred_w = df["pred_fused"].values
y_prob_w = df["prob_robot_fused"].values

print_metrics(
    title=f"WINDOW-LEVEL WEIGHTED FUSION: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_w,
    y_pred=y_pred_w,
    y_prob=y_prob_w,
)

print("\nWindow-level preview:")
print(df[window_cols].head(30).to_string(index=False))


# ---------------------------------------------------------------------
# RUN-LEVEL AGGREGATION FROM WINDOW-LEVEL FUSION
# ---------------------------------------------------------------------

run_rows = []

for_run = df.groupby("held_out_run")

for held_out_run, group in for_run:
    true_label = int(group["true_label"].iloc[0])

    prob_robot_fnirs_mean = float(group["prob_robot_fnirs"].mean())
    prob_robot_emg_mean = float(group["prob_robot_emg"].mean())
    prob_robot_fused_mean = float(group["prob_robot_fused"].mean())

    pred_fused_run = int(prob_robot_fused_mean >= THRESHOLD)

    run_rows.append({
        "held_out_run": held_out_run,
        "true_label": true_label,
        "prob_robot_fnirs_mean": prob_robot_fnirs_mean,
        "prob_robot_emg_mean": prob_robot_emg_mean,
        "prob_robot_fused_mean": prob_robot_fused_mean,
        "pred_fused_run": pred_fused_run,
        "correct_fused_run": pred_fused_run == true_label,
        "n_windows": int(len(group)),
        "window_accuracy_within_run": float(group["correct_fused"].mean()),
    })

run_df = pd.DataFrame(run_rows).sort_values("held_out_run")
run_df.to_csv(RUN_OUT, index=False)

y_true_r = run_df["true_label"].values
y_pred_r = run_df["pred_fused_run"].values
y_prob_r = run_df["prob_robot_fused_mean"].values

print_metrics(
    title=f"RUN-LEVEL WEIGHTED FUSION FROM WINDOW MEANS: EMG={W_EMG}, fNIRS={W_FNIRS}, threshold={THRESHOLD}",
    y_true=y_true_r,
    y_pred=y_pred_r,
    y_prob=y_prob_r,
)

print("\nRun-level predictions:")
print(run_df.to_string(index=False))

print("\nSaved:")
print(f"  Window-level: {WINDOW_OUT}")
print(f"  Run-level:    {RUN_OUT}")
