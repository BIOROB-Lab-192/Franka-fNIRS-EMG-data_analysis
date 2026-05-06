import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

EMG_PATH = "./machine_learning/EMG_pertask/export/predictions_task_level_loocv.csv"
FNIRS_PATH = "./machine_learning/fNIRS_pertask/export/predictions_task_level_loocv.csv"

W_FNIRS = 0.3
W_EMG = 0.7

emg = pd.read_csv(EMG_PATH)
fnirs = pd.read_csv(FNIRS_PATH)

prob_cols = [c for c in emg.columns if c.startswith("prob_")]

emg_small = emg[["held_out_run", "task_instance", "true_label", "true_task"] + prob_cols].copy()
fnirs_small = fnirs[["held_out_run", "task_instance", "true_label", "true_task"] + prob_cols].copy()

df = emg_small.merge(
    fnirs_small,
    on=["held_out_run", "task_instance"],
    suffixes=("_emg", "_fnirs"),
)

if not (df["true_label_emg"] == df["true_label_fnirs"]).all():
    raise ValueError("EMG and fNIRS true labels do not match.")

df["true_label"] = df["true_label_emg"]
df["true_task"] = df["true_task_emg"]

task_names = [c.replace("prob_", "") for c in prob_cols]

for task in task_names:
    df[f"prob_{task}_fused"] = (
        W_EMG * df[f"prob_{task}_emg"]
        + W_FNIRS * df[f"prob_{task}_fnirs"]
    )

fused_prob_cols = [f"prob_{task}_fused" for task in task_names]

df["pred_label_fused"] = df[fused_prob_cols].values.argmax(axis=1)
df["pred_task_fused"] = [task_names[i] for i in df["pred_label_fused"]]
df["correct_fused"] = df["pred_label_fused"] == df["true_label"]

print("Accuracy:", accuracy_score(df["true_label"], df["pred_label_fused"]))
print("Balanced accuracy:", balanced_accuracy_score(df["true_label"], df["pred_label_fused"]))
print("Macro F1:", f1_score(df["true_label"], df["pred_label_fused"], average="macro"))
print()
print(classification_report(df["true_label"], df["pred_label_fused"], target_names=task_names))

df.to_csv("./machine_learning/task_late_fusion_simple.csv", index=False)