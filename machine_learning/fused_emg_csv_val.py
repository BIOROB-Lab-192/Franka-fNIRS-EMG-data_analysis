import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

FNIRS_PATH = "./machine_learning/fNIRS_perrun/export/predictions_run_level_loocv.csv"
EMG_PATH = "./machine_learning/EMG_perrun/export/predictions_run_level_loocv.csv"

fnirs = pd.read_csv(FNIRS_PATH)
emg = pd.read_csv(EMG_PATH)

fnirs = fnirs[["held_out_run", "true_label", "prob_robot_mean"]].rename(
    columns={"prob_robot_mean": "prob_robot_fnirs"}
)

emg = emg[["held_out_run", "true_label", "prob_robot_mean"]].rename(
    columns={"prob_robot_mean": "prob_robot_emg"}
)

df = fnirs.merge(emg, on="held_out_run", suffixes=("_fnirs", "_emg"))

if not np.all(df["true_label_fnirs"] == df["true_label_emg"]):
    raise ValueError("Labels do not match between fNIRS and EMG files.")

df["true_label"] = df["true_label_fnirs"]

# Primary, pre-specified fusion rule
W_FNIRS = 0.2
W_EMG = 0.8

df["prob_robot_fused"] = (
    W_FNIRS * df["prob_robot_fnirs"]
    + W_EMG * df["prob_robot_emg"]
)

df["pred_fused"] = (df["prob_robot_fused"] >= 0.5).astype(int)
df["correct_fused"] = df["pred_fused"] == df["true_label"]

y_true = df["true_label"].values
y_pred = df["pred_fused"].values
y_prob = df["prob_robot_fused"].values

print("Equal-weight fusion")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision robot:", precision_score(y_true, y_pred, pos_label=1, zero_division=0))
print("Recall robot:", recall_score(y_true, y_pred, pos_label=1, zero_division=0))
print("F1 robot:", f1_score(y_true, y_pred, pos_label=1, zero_division=0))
print("AUC:", roc_auc_score(y_true, y_prob))
print()
print(df[[
    "held_out_run",
    "true_label",
    "prob_robot_fnirs",
    "prob_robot_emg",
    "prob_robot_fused",
    "pred_fused",
    "correct_fused",
]])

df.to_csv("./machine_learning/fusion_run_level_loocv_equal_weights.csv", index=False)