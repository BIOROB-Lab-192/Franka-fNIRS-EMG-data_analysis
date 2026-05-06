from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ──────────────────────────── CONFIG ────────────────────────────

EMG_PATH = Path("./data/processed/combined/data_packet/emg_rms.parquet")
FNIRS_PATH = Path("./data/processed/combined/data_packet/fnirs_full.parquet")

RUN_COL = "run_id"
INSTANCE_COL = "task_instance"
LABEL_COL = "is_robot"

EMG_COLUMNS = [
    "Avanti Sensor 1 (82703) | EMG 1 (mV)",
    "Avanti Sensor 2 (82529) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 1 (mV)",
    "Duo Sensor 3 (78042) | EMG 2 (mV)",
]

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


# ──────────────────────────── FEATURE EXTRACTION ────────────────────────────


def safe_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def channel_features(x, prefix, col_names):
    """
    x shape: T x C
    Returns one feature dict for the whole task window.
    """
    x = safe_np(x)
    feats = {}

    for i, name in enumerate(col_names):
        sig = x[:, i]
        base = f"{prefix}_{i}"

        feats[f"{base}_mean"] = np.mean(sig)
        feats[f"{base}_std"] = np.std(sig)
        feats[f"{base}_min"] = np.min(sig)
        feats[f"{base}_max"] = np.max(sig)
        feats[f"{base}_median"] = np.median(sig)
        feats[f"{base}_p25"] = np.percentile(sig, 25)
        feats[f"{base}_p75"] = np.percentile(sig, 75)
        feats[f"{base}_range"] = np.max(sig) - np.min(sig)
        feats[f"{base}_auc"] = np.trapezoid(sig)

        # Early vs late difference
        n = len(sig)
        third = max(1, n // 3)
        early = np.mean(sig[:third])
        late = np.mean(sig[-third:])
        feats[f"{base}_late_minus_early"] = late - early

        # Simple linear slope
        if n > 1:
            t = np.arange(n)
            slope = np.polyfit(t, sig, 1)[0]
        else:
            slope = 0.0
        feats[f"{base}_slope"] = slope

    return feats


def extract_features_from_df(df, signal_cols, prefix, time_sort_col):
    rows = []

    run_ids = df[RUN_COL].unique().sort().to_list()

    for run_id in run_ids:
        run_df = df.filter(pl.col(RUN_COL) == run_id)
        label = int(run_df[LABEL_COL][0])

        instances = run_df[INSTANCE_COL].unique().sort().to_list()

        for inst in instances:
            inst_df = run_df.filter(pl.col(INSTANCE_COL) == inst).sort(time_sort_col)

            x = inst_df.select(signal_cols).to_numpy()

            feats = {
                "run_id": run_id,
                "task_instance": int(inst),
                "is_robot": label,
            }

            feats.update(channel_features(x, prefix, signal_cols))
            rows.append(feats)

    return pd.DataFrame(rows)


def load_feature_tables():
    print("Loading EMG...")
    emg_df = pl.read_parquet(EMG_PATH)

    print("Loading fNIRS...")
    fnirs_df = pl.read_parquet(FNIRS_PATH)

    print("Extracting EMG features...")
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

    print("EMG feature table:", emg_feat.shape)
    print("fNIRS feature table:", fnirs_feat.shape)

    fused = emg_feat.merge(
        fnirs_feat,
        on=["run_id", "task_instance", "is_robot"],
        how="inner",
    )

    print("Combined feature table:", fused.shape)

    return emg_feat, fnirs_feat, fused


# ──────────────────────────── LOOCV ────────────────────────────


def evaluate_loocv(feature_df, name):
    print("\n" + "=" * 80)
    print(f"Evaluating: {name}")
    print("=" * 80)

    run_ids = sorted(feature_df["run_id"].unique())

    feature_cols = [
        c
        for c in feature_df.columns
        if c not in ["run_id", "task_instance", "is_robot"]
    ]

    models = {
        "small_mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 16),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        batch_size=8,
                        learning_rate_init=5e-4,
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=30,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "small_mlp_no_earlystop": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 8),
                        activation="relu",
                        solver="adam",
                        alpha=5e-3,
                        batch_size=8,
                        learning_rate_init=3e-4,
                        max_iter=500,
                        early_stopping=False,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_l2": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=0.5,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logreg_l1": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=0.2,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="linear",
                        C=0.5,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RidgeClassifier(
                        alpha=10.0,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        ),
    }

    all_results = []

    for model_name, model in models.items():
        all_true = []
        all_pred = []
        all_prob = []
        run_rows = []

        for held_out in run_ids:
            train = feature_df[feature_df["run_id"] != held_out].copy()
            test = feature_df[feature_df["run_id"] == held_out].copy()

            X_train = train[feature_cols].to_numpy()
            y_train = train["is_robot"].astype(int).to_numpy()

            X_test = test[feature_cols].to_numpy()
            y_test = test["is_robot"].astype(int).to_numpy()

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_test)[:, 1]
            else:
                # RidgeClassifier does not have predict_proba.
                # Use decision_function and squash roughly to 0-1.
                scores = model.decision_function(X_test)
                prob = 1 / (1 + np.exp(-scores))

            all_true.extend(y_test)
            all_pred.extend(pred)
            all_prob.extend(prob)

            # Run-level aggregation
            run_true = int(y_test[0])
            run_prob = float(np.mean(prob))
            run_pred = int(run_prob >= 0.5)
            run_correct = run_pred == run_true

            run_rows.append(
                {
                    "held_out_run": held_out,
                    "true_label": run_true,
                    "prob_robot_mean": run_prob,
                    "pred_label": run_pred,
                    "correct": run_correct,
                    "n_windows": len(y_test),
                    "window_accuracy": accuracy_score(y_test, pred),
                }
            )

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        all_prob = np.array(all_prob)

        run_df = pd.DataFrame(run_rows)
        run_true = run_df["true_label"].to_numpy()
        run_pred = run_df["pred_label"].to_numpy()
        run_prob = run_df["prob_robot_mean"].to_numpy()

        try:
            window_auc = roc_auc_score(all_true, all_prob)
        except Exception:
            window_auc = np.nan

        try:
            run_auc = roc_auc_score(run_true, run_prob)
        except Exception:
            run_auc = np.nan

        result = {
            "dataset": name,
            "model": model_name,
            "window_accuracy": accuracy_score(all_true, all_pred),
            "window_balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
            "window_f1_robot": f1_score(
                all_true, all_pred, pos_label=1, zero_division=0
            ),
            "window_auc": window_auc,
            "run_accuracy": accuracy_score(run_true, run_pred),
            "run_balanced_accuracy": balanced_accuracy_score(run_true, run_pred),
            "run_f1_robot": f1_score(run_true, run_pred, pos_label=1, zero_division=0),
            "run_auc": run_auc,
        }

        all_results.append(result)

        print(f"\n{name} | {model_name}")
        print("-" * 60)
        print(f"Window acc: {result['window_accuracy']:.3f}")
        print(f"Window balanced acc: {result['window_balanced_accuracy']:.3f}")
        print(f"Window F1_robot: {result['window_f1_robot']:.3f}")
        print(f"Window AUC: {result['window_auc']:.3f}")
        print(f"Run acc: {result['run_accuracy']:.3f}")
        print(f"Run balanced acc: {result['run_balanced_accuracy']:.3f}")
        print(f"Run F1_robot: {result['run_f1_robot']:.3f}")
        print(f"Run AUC: {result['run_auc']:.3f}")

        print("\nRun-level predictions:")
        print(run_df.to_string(index=False))

        print("\nRun-level confusion matrix:")
        print(confusion_matrix(run_true, run_pred, labels=[0, 1]))

    results_df = pd.DataFrame(all_results)
    return results_df


# ──────────────────────────── RUN EXPERIMENT ────────────────────────────

emg_feat, fnirs_feat, fused_feat = load_feature_tables()

results = pd.concat(
    [
        evaluate_loocv(emg_feat, "EMG_features"),
        evaluate_loocv(fnirs_feat, "fNIRS_features"),
        evaluate_loocv(fused_feat, "EMG_plus_fNIRS_features"),
    ],
    ignore_index=True,
)

print("\n" + "=" * 80)
print("SUMMARY SORTED BY RUN ACCURACY")
print("=" * 80)
print(
    results.sort_values(
        ["run_accuracy", "run_f1_robot", "window_accuracy"], ascending=False
    ).to_string(index=False)
)

print("\nSUMMARY SORTED BY WINDOW ACCURACY")
print("=" * 80)
print(
    results.sort_values(
        ["window_accuracy", "window_f1_robot", "run_accuracy"], ascending=False
    ).to_string(index=False)
)

top = results.sort_values(
    ["run_accuracy", "run_auc", "window_accuracy"], ascending=False
)
print("\n")
print(top.head(5).to_string(index=False))
