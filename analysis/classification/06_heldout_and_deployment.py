
from __future__ import annotations

# Configure paths via tools/run_analysis.py or NADES_DATA_ROOT.
if "data_path" not in globals():
    import os as _path_os
    import sys as _path_sys
    from pathlib import Path as _Path
    _path_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "tools"))
    from project_paths import bind_paths as _bind_paths
    if not _path_os.environ.get("NADES_DATA_ROOT"):
        raise RuntimeError("Run through tools/run_analysis.py --data-root <Curated>, or set NADES_DATA_ROOT.")
    data_path = _bind_paths(_path_os.environ["NADES_DATA_ROOT"])


# ============================================================
# FINAL60 INTERNAL EVALUATION + DEPLOYMENT REFIT
#
# Clean workflow:
#   1. Reuse the split created before supervised prefiltering.
#   2. Load the 60 descriptors selected by train-only DAGA.
#   3. Fit imputation and XGBoost on the 511 training samples only.
#   4. Evaluate once on the fixed 128-sample internal-evaluation set.
#   5. Refit on all 639 labelled samples only after evaluation.
#
# Blind data are deliberately not loaded here.
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# 0. PATHS
# ============================================================

BASE = data_path('')

TRAIN_PATH = data_path('master_train_features.csv')
CANDIDATE_PATH = data_path('GA_candidate_descriptors_1000.csv')
SPLIT_IN = data_path('Final_clean_split_seed42.csv')

OUTPUT_ROOT = data_path('DiversityGA_SubsetSize_Comparison')
SELECTED_FEATURES_PATH = (
    data_path('DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('Final60_Internal_Evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PREDICTIONS_OUT = data_path('Final60_Internal_Evaluation/Final60_train_predictions.csv')
EVALUATION_PREDICTIONS_OUT = (
    data_path('Final60_Internal_Evaluation/Final60_internal_evaluation_predictions.csv')
)
METRICS_OUT = data_path('Final60_Internal_Evaluation/Final60_train_internal_evaluation_metrics.csv')
FEATURES_OUT = data_path('Final60_Internal_Evaluation/Final60_selected_descriptors.csv')
IMPORTANCE_OUT = data_path('Final60_Internal_Evaluation/Final60_feature_importance.csv')
ROC_CURVE_OUT = data_path('Final60_Internal_Evaluation/Final60_ROC_train_test.png')
PR_CURVE_OUT = data_path('Final60_Internal_Evaluation/Final60_PR_train_test.png')
EVALUATION_MODEL_OUT = (
    data_path('Final60_Internal_Evaluation/Final60_train_subset_model_bundle.joblib')
)
EVALUATION_BOOSTER_OUT = (
    data_path('Final60_Internal_Evaluation/Final60_train_subset_xgboost_model.json')
)
FINAL_MODEL_OUT = (
    data_path('Final60_Internal_Evaluation/Final60_full_dataset_model_bundle.joblib')
)
FINAL_BOOSTER_OUT = (
    data_path('Final60_Internal_Evaluation/Final60_full_dataset_xgboost_model.json')
)


# ============================================================
# 1. FROZEN SETTINGS
# ============================================================

SPLIT_SEED = 42
MODEL_SEED = 82
TARGET_COLUMN = "activity"
CODE_COLUMN = "Code"
EXPECTED_FEATURE_COUNT = 60

# The OOF-derived operating threshold for the selected Final60 model.
# Do not change this after inspecting internal-evaluation or blind predictions.
THRESHOLD = 0.42

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "random_state": MODEL_SEED,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

plt.rcParams["font.family"] = "Arial"

# ============================================================
# 2. HELPERS
# ============================================================

def validate_binary_target(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} contains missing or non-numeric values.")

    unexpected = set(values.unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"{name} must contain only 0 and 1: {unexpected}")

    return values.astype(np.int8).to_numpy()


def calculate_scale_pos_weight(y_values: np.ndarray) -> float:
    n_non_nades = int(np.sum(y_values == 0))
    n_nades = int(np.sum(y_values == 1))
    if n_non_nades == 0 or n_nades == 0:
        raise ValueError("Both target classes must be present.")
    return n_non_nades / n_nades


def build_model(seed: int, scale_pos_weight: float) -> xgb.XGBClassifier:
    parameters = dict(XGB_PARAMS)
    parameters["random_state"] = seed
    return xgb.XGBClassifier(
        **parameters,
        scale_pos_weight=scale_pos_weight,
    )


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "n_samples": int(len(y_true)),
        "n_non_NADES": int(np.sum(y_true == 0)),
        "n_NADES": int(np.sum(y_true == 1)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision_NADES": float(
            average_precision_score(y_true, probability)
        ),
        "average_precision_non_NADES": float(
            average_precision_score(1 - y_true, 1.0 - probability)
        ),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "specificity_non_NADES": float(specificity),
        "false_positive_rate": float(fpr),
        "precision_NADES": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall_NADES": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "f1_NADES": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_prediction_table(
    codes: pd.Series,
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    split_name: str,
) -> pd.DataFrame:
    prediction = (probability >= threshold).astype(np.int8)
    return pd.DataFrame(
        {
            "Code": codes.reset_index(drop=True),
            "split": split_name,
            "activity": y_true,
            "true_class": np.where(y_true == 1, "NADES", "non-NADES"),
            "probability_NADES": probability,
            "probability_non_NADES": 1.0 - probability,
            "threshold_used": threshold,
            "predicted_activity": prediction,
            "predicted_class": np.where(
                prediction == 1,
                "NADES",
                "non-NADES",
            ),
            "correct_prediction": prediction == y_true,
            "signed_margin": probability - threshold,
            "absolute_margin": np.abs(probability - threshold),
        }
    )


def print_results(name: str, y_true: np.ndarray, probability: np.ndarray) -> dict:
    metrics = calculate_metrics(y_true, probability, THRESHOLD)
    prediction = (probability >= THRESHOLD).astype(np.int8)

    print("\n" + "=" * 96)
    print(f"{name.upper()} RESULTS")
    print("=" * 96)
    for label, key in [
        ("Samples", "n_samples"),
        ("ROC-AUC", "roc_auc"),
        ("Average precision (NADES)", "average_precision_NADES"),
        ("Average precision (non-NADES)", "average_precision_non_NADES"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("MCC", "mcc"),
        ("Specificity (non-NADES)", "specificity_non_NADES"),
        ("Recall (NADES)", "recall_NADES"),
        ("Precision (NADES)", "precision_NADES"),
        ("F1 (NADES)", "f1_NADES"),
    ]:
        value = metrics[key]
        if isinstance(value, (int, np.integer)):
            print(f"{label:30s}: {value}")
        else:
            print(f"{label:30s}: {value:.4f}")

    print(f"Operating threshold             : {THRESHOLD:.2f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, prediction, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            prediction,
            target_names=["non-NADES", "NADES"],
            digits=4,
            zero_division=0,
        )
    )
    return metrics


# ============================================================
# 3. LOAD AND VALIDATE INPUTS
# ============================================================

for required_path in [
    TRAIN_PATH,
    CANDIDATE_PATH,
    SPLIT_IN,
    SELECTED_FEATURES_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found:\n{required_path}")

data = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)
candidate_table = pd.read_csv(CANDIDATE_PATH, low_memory=False)
selected_table = pd.read_csv(SELECTED_FEATURES_PATH, low_memory=False)
split_table = pd.read_csv(SPLIT_IN, dtype={CODE_COLUMN: "string"})

if not {CODE_COLUMN, TARGET_COLUMN}.issubset(data.columns):
    raise ValueError("Training data must contain Code and activity columns.")
if "feature" not in candidate_table.columns:
    raise ValueError("Candidate table must contain a feature column.")
if "feature" not in selected_table.columns:
    raise ValueError("Selected-descriptor table must contain a feature column.")
if not {CODE_COLUMN, TARGET_COLUMN, "split"}.issubset(split_table.columns):
    raise ValueError("Saved split must contain Code, activity and split columns.")

selected_features = (
    selected_table["feature"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

if len(selected_features) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected exactly {EXPECTED_FEATURE_COUNT} descriptors; "
        f"found {len(selected_features)}."
    )

candidate_features = set(
    candidate_table["feature"].dropna().astype(str).str.strip()
)
not_in_candidate_pool = [
    feature for feature in selected_features if feature not in candidate_features
]
if not_in_candidate_pool:
    raise ValueError(
        "Selected descriptors are not present in the train-only candidate pool: "
        f"{not_in_candidate_pool[:20]}"
    )

codes = data[CODE_COLUMN].astype("string").str.strip()
if codes.isna().any() or codes.duplicated().any():
    raise ValueError("Training Code values must be complete and unique.")

y_all = validate_binary_target(data[TARGET_COLUMN], TARGET_COLUMN)

split_table[CODE_COLUMN] = split_table[CODE_COLUMN].astype("string").str.strip()
if split_table[CODE_COLUMN].isna().any() or split_table[CODE_COLUMN].duplicated().any():
    raise ValueError("Saved split Code values must be complete and unique.")

split_codes = set(split_table[CODE_COLUMN])
data_codes = set(codes)
if split_codes != data_codes:
    raise ValueError(
        "Saved split and training data do not contain the same Code set."
    )

split_activity = pd.to_numeric(split_table[TARGET_COLUMN], errors="coerce")
activity_by_code = dict(zip(split_table[CODE_COLUMN], split_activity))
if not np.array_equal(
    y_all,
    np.asarray([activity_by_code[code] for code in codes], dtype=np.int8),
):
    raise ValueError("Saved split activity labels do not match current training data.")

split_by_code = dict(zip(split_table[CODE_COLUMN], split_table["split"]))
split_labels = codes.map(split_by_code)
allowed_split_labels = {"Train", "Internal_evaluation"}
if set(split_labels.unique()) != allowed_split_labels:
    raise ValueError(
        "Saved split must contain exactly Train and Internal_evaluation labels. "
        f"Found: {sorted(set(split_labels.unique()))}"
    )

train_mask = (split_labels == "Train").to_numpy()
evaluation_mask = (split_labels == "Internal_evaluation").to_numpy()

if train_mask.sum() == 0 or evaluation_mask.sum() == 0:
    raise ValueError("Saved split contains an empty partition.")

missing_features = [
    feature for feature in selected_features if feature not in data.columns
]
if missing_features:
    raise ValueError(
        "Selected descriptors missing from training data: "
        f"{missing_features[:20]}"
    )

X_all_raw = (
    data[selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)

X_train_raw = X_all_raw.loc[train_mask].reset_index(drop=True)
X_evaluation_raw = X_all_raw.loc[evaluation_mask].reset_index(drop=True)
y_train = y_all[train_mask]
y_evaluation = y_all[evaluation_mask]
codes_train = codes.loc[train_mask].reset_index(drop=True)
codes_evaluation = codes.loc[evaluation_mask].reset_index(drop=True)

print("=" * 96)
print("FINAL60 CLEAN INTERNAL EVALUATION")
print("=" * 96)
print("Split source                 :", SPLIT_IN.name)
print("Selected-descriptor source   :", SELECTED_FEATURES_PATH)
print("Training samples             :", len(y_train))
print("Internal-evaluation samples  :", len(y_evaluation))
print("Training class counts        :", dict(zip(*np.unique(y_train, return_counts=True))))
print("Evaluation class counts      :", dict(zip(*np.unique(y_evaluation, return_counts=True))))
print("Selected descriptors         :", len(selected_features))
print(f"Frozen operating threshold   : {THRESHOLD:.2f}")


# ============================================================
# 4. INTERNAL EVALUATION — FIT ON TRAIN ONLY
# ============================================================

evaluation_imputer = SimpleImputer(strategy="median")
X_train = evaluation_imputer.fit_transform(X_train_raw).astype(np.float32)
X_evaluation = evaluation_imputer.transform(X_evaluation_raw).astype(np.float32)

train_spw = calculate_scale_pos_weight(y_train)
evaluation_model = build_model(MODEL_SEED, train_spw)
evaluation_model.fit(X_train, y_train)

train_probability = evaluation_model.predict_proba(X_train)[:, 1]
evaluation_probability = evaluation_model.predict_proba(X_evaluation)[:, 1]

train_metrics = print_results("Training resubstitution diagnostic", y_train, train_probability)
evaluation_metrics = print_results(
    "Fixed internal-evaluation subset",
    y_evaluation,
    evaluation_probability,
)

make_prediction_table(
    codes_train,
    y_train,
    train_probability,
    THRESHOLD,
    "Train",
).to_csv(TRAIN_PREDICTIONS_OUT, index=False)

make_prediction_table(
    codes_evaluation,
    y_evaluation,
    evaluation_probability,
    THRESHOLD,
    "Internal_evaluation",
).to_csv(EVALUATION_PREDICTIONS_OUT, index=False)

pd.DataFrame(
    [
        {"dataset": "Train_resubstitution", **train_metrics},
        {"dataset": "Internal_evaluation", **evaluation_metrics},
    ]
).to_csv(METRICS_OUT, index=False)

selected_table.to_csv(FEATURES_OUT, index=False)

importance = pd.DataFrame(
    {
        "feature": selected_features,
        "importance_gain": evaluation_model.feature_importances_,
    }
).sort_values("importance_gain", ascending=False).reset_index(drop=True)
importance.insert(0, "rank", np.arange(1, len(importance) + 1))
importance.to_csv(IMPORTANCE_OUT, index=False)


# ============================================================
# 5. ROC CURVES — TRAIN VS TEST
#
# The plotting style is retained from the original Final script.
# Here, "Test" is the pre-saved fixed internal-evaluation subset.
# ============================================================

y_test = y_evaluation
test_probability = evaluation_probability
test_metrics = evaluation_metrics

fpr_train, tpr_train, _ = roc_curve(
    y_train,
    train_probability,
)

fpr_test, tpr_test, _ = roc_curve(
    y_test,
    test_probability,
)

plt.figure(figsize=(10, 10))

plt.plot(
    fpr_train,
    tpr_train,
    linewidth=3,
    label=(
        f"Train (AUC = "
        f"{train_metrics['roc_auc']:.3f})"
    ),
)

plt.plot(
    fpr_test,
    tpr_test,
    linewidth=3,
    label=(
        f"Test (AUC = "
        f"{test_metrics['roc_auc']:.3f})"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
)

plt.title("")
plt.xlabel("False Positive Rate", fontsize=28)
plt.ylabel("True Positive Rate", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=20)
plt.grid(False)
plt.tight_layout()

plt.savefig(
    ROC_CURVE_OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

print(f"Saved: {ROC_CURVE_OUT}")


# ============================================================
# 6. PRECISION–RECALL CURVES — TRAIN VS TEST
# ============================================================

precision_train, recall_train, _ = precision_recall_curve(
    y_train,
    train_probability,
)

precision_test, recall_test, _ = precision_recall_curve(
    y_test,
    test_probability,
)

plt.figure(figsize=(10, 10))

plt.plot(
    recall_train,
    precision_train,
    linewidth=3,
    label=(
        f"Train (AP = "
        f"{train_metrics['average_precision_NADES']:.3f})"
    ),
)

plt.plot(
    recall_test,
    precision_test,
    linewidth=3,
    label=(
        f"Test (AP = "
        f"{test_metrics['average_precision_NADES']:.3f})"
    ),
)

plt.title("")
plt.xlabel("Recall", fontsize=28)
plt.ylabel("Precision", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=20)
plt.grid(False)
plt.tight_layout()

plt.savefig(
    PR_CURVE_OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

print(f"Saved: {PR_CURVE_OUT}")


# ============================================================
# 7. FINAL DEPLOYMENT REFIT ON ALL LABELLED DATA
#
# This happens only after internal evaluation is completed.
# The resulting bundle is for the separate blind-prediction script.
# ============================================================

final_imputer = SimpleImputer(strategy="median")
X_full = final_imputer.fit_transform(X_all_raw).astype(np.float32)
full_spw = calculate_scale_pos_weight(y_all)

final_model = build_model(MODEL_SEED, full_spw)
final_model.fit(X_full, y_all)

evaluation_bundle = {
    "model": evaluation_model,
    "imputer": evaluation_imputer,
    "selected_features": selected_features,
    "threshold": THRESHOLD,
    "split_source": data_path('Final_clean_split_seed42.csv'),
    "selected_features_source": data_path('DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv'),
    "xgb_parameters": XGB_PARAMS,
    "scale_pos_weight": train_spw,
    "purpose": "Evaluation model fitted only on the training partition.",
}
joblib.dump(evaluation_bundle, EVALUATION_MODEL_OUT)
evaluation_model.save_model(EVALUATION_BOOSTER_OUT)

final_bundle = {
    "model": final_model,
    "imputer": final_imputer,
    "selected_features": selected_features,
    "threshold": THRESHOLD,
    "selected_features_source": data_path('DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv'),
    "xgb_parameters": XGB_PARAMS,
    "scale_pos_weight": full_spw,
    "training_samples": int(len(y_all)),
    "purpose": "Deployment model refitted on all labelled samples after internal evaluation.",
}
joblib.dump(final_bundle, FINAL_MODEL_OUT)
final_model.save_model(FINAL_BOOSTER_OUT)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 96)
print("FINAL60 INTERNAL-EVALUATION SUMMARY")
print("=" * 96)
print(f"Frozen threshold             : {THRESHOLD:.2f}")
print(f"Internal-evaluation ROC-AUC  : {evaluation_metrics['roc_auc']:.4f}")
print(
    "Internal-evaluation AP NADES : "
    f"{evaluation_metrics['average_precision_NADES']:.4f}"
)
print(f"Balanced accuracy            : {evaluation_metrics['balanced_accuracy']:.4f}")
print(f"MCC                          : {evaluation_metrics['mcc']:.4f}")
print(f"Specificity (non-NADES)      : {evaluation_metrics['specificity_non_NADES']:.4f}")
print(f"Recall (NADES)               : {evaluation_metrics['recall_NADES']:.4f}")
print("\nSaved:")
for output_path in [
    TRAIN_PREDICTIONS_OUT,
    EVALUATION_PREDICTIONS_OUT,
    METRICS_OUT,
    FEATURES_OUT,
    IMPORTANCE_OUT,
    ROC_CURVE_OUT,
    PR_CURVE_OUT,
    EVALUATION_MODEL_OUT,
    EVALUATION_BOOSTER_OUT,
    FINAL_MODEL_OUT,
    FINAL_BOOSTER_OUT,
]:
    print(" -", output_path)

print("\nBlind data were not loaded or used.")


