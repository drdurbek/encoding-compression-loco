
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
# FINAL60 BLIND-SET PREDICTION
#
# This script does NOT fit, tune, select features, or change tau.
# It loads the frozen Final60 full-dataset deployment bundle and
# applies it to the blind data. If blind labels are present, they
# are used only after prediction for post hoc external evaluation.
#
# Required output from Final60_Internal_Evaluation.py:
#   Final60_full_dataset_model_bundle.joblib
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb  # Required for safe XGBoost bundle deserialization.

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. PATHS
# ============================================================

BASE = data_path('')

BLIND_PATH = data_path('master_blind_features.csv')

# This bundle was fitted on all 639 labelled mixtures only after
# internal evaluation had been completed and recorded.
FINAL_MODEL_PATH = (
    data_path('Final60_Internal_Evaluation/Final60_full_dataset_model_bundle.joblib')
)

BLIND_PREDICTIONS_OUT = data_path('Final60_blind_predictions.csv')
BLIND_SUMMARY_OUT = data_path('Final60_blind_prediction_summary.csv')


# ============================================================
# 2. INPUT SETTINGS
# ============================================================

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

EXPECTED_FEATURE_COUNT = 60


# ============================================================
# 3. HELPERS
# ============================================================

def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity for class 0 = non-NADES."""
    tn, fp, _, _ = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return float(tn / (tn + fp)) if (tn + fp) else np.nan


def print_metric(name: str, value: float) -> None:
    """Print a numeric metric or NA cleanly."""
    if pd.isna(value):
        print(f"{name:<30}: NA")
    else:
        print(f"{name:<30}: {value:.4f}")


# ============================================================
# 4. LOAD FROZEN MODEL BUNDLE AND BLIND DATA
# ============================================================

if not BLIND_PATH.exists():
    raise FileNotFoundError(
        f"Blind feature file was not found:\n{BLIND_PATH}"
    )

if not FINAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        "The frozen Final60 deployment bundle was not found:\n"
        f"{FINAL_MODEL_PATH}\n\n"
        "Run Final60_Internal_Evaluation.py first."
    )

bundle = joblib.load(FINAL_MODEL_PATH)

required_bundle_keys = {
    "model",
    "imputer",
    "selected_features",
    "threshold",
}

missing_bundle_keys = required_bundle_keys - set(bundle)
if missing_bundle_keys:
    raise ValueError(
        "The Final60 model bundle is missing: "
        f"{sorted(missing_bundle_keys)}"
    )

final_model = bundle["model"]
imputer = bundle["imputer"]
selected_features = list(bundle["selected_features"])
THRESHOLD = float(bundle["threshold"])

if len(selected_features) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        "The frozen model bundle must contain exactly "
        f"{EXPECTED_FEATURE_COUNT} descriptors; found "
        f"{len(selected_features)}."
    )

blind = pd.read_csv(
    BLIND_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

print("=" * 94)
print("FINAL60 BLIND-SET PREDICTION")
print("=" * 94)
print("Model bundle         :", FINAL_MODEL_PATH)
print("Blind feature table  :", BLIND_PATH)
print("Blind shape          :", blind.shape)
print("Fixed descriptors    :", len(selected_features))
print(f"Frozen threshold     : {THRESHOLD:.2f}")


# ============================================================
# 5. VALIDATE THE BLIND INPUT
# ============================================================

if "Code" not in blind.columns:
    raise ValueError("master_blind_features.csv must contain 'Code'.")

blind_codes = blind["Code"].astype("string").str.strip()

if blind_codes.isna().any():
    raise ValueError("Blind data contain missing Code values.")

if blind_codes.duplicated().any():
    duplicate_codes = blind_codes[
        blind_codes.duplicated(keep=False)
    ].tolist()
    raise ValueError(
        "Duplicate Code values in blind data. Examples: "
        f"{duplicate_codes[:10]}"
    )

missing_features = [
    feature
    for feature in selected_features
    if feature not in blind.columns
]

if missing_features:
    raise ValueError(
        "Selected Final60 descriptors are missing from blind data: "
        f"{missing_features}"
    )

if hasattr(imputer, "n_features_in_"):
    if imputer.n_features_in_ != len(selected_features):
        raise ValueError(
            "The frozen imputer expects "
            f"{imputer.n_features_in_} features, but the bundle contains "
            f"{len(selected_features)} selected descriptors."
        )


# ============================================================
# 6. TRANSFORM BLIND DATA WITH THE FROZEN TRAINED IMPUTER
# ============================================================

X_blind_raw = (
    blind.loc[:, selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)

# transform only: no fit is performed on the blind data.
X_blind = imputer.transform(X_blind_raw).astype(np.float32)

print("Blind model matrix   :", X_blind.shape)


# ============================================================
# 7. PREDICT THE BLIND SET
# ============================================================

blind_probability_nades = final_model.predict_proba(X_blind)[:, 1]

blind_prediction = (
    blind_probability_nades >= THRESHOLD
).astype(np.int8)

blind_output = pd.DataFrame(
    {
        "Code": blind_codes.to_numpy(),
        "probability_NADES": blind_probability_nades,
        "probability_non_NADES": 1.0 - blind_probability_nades,
        "threshold_used": THRESHOLD,
        "predicted_activity": blind_prediction,
        "predicted_class": np.where(
            blind_prediction == 1,
            "NADES",
            "non-NADES",
        ),
        "signed_margin": blind_probability_nades - THRESHOLD,
        "absolute_margin": np.abs(
            blind_probability_nades - THRESHOLD
        ),
    }
)


# ============================================================
# 8. OPTIONAL POST HOC EXTERNAL EVALUATION
#
# A blind activity column, if present, is read only here. It is
# never used in feature selection, imputation, model fitting, or
# threshold definition.
# ============================================================

blind_metric_values = {
    "blind_labeled_samples": 0,
    "blind_n_NADES": np.nan,
    "blind_n_non_NADES": np.nan,
    "blind_roc_auc": np.nan,
    "blind_pr_auc_NADES": np.nan,
    "blind_pr_auc_non_NADES": np.nan,
    "blind_balanced_accuracy": np.nan,
    "blind_mcc": np.nan,
    "blind_precision_NADES": np.nan,
    "blind_recall_NADES": np.nan,
    "blind_f1_NADES": np.nan,
    "blind_specificity_non_NADES": np.nan,
    "blind_tn": np.nan,
    "blind_fp": np.nan,
    "blind_fn": np.nan,
    "blind_tp": np.nan,
}

if "activity" in blind.columns:
    blind_activity = pd.to_numeric(
        blind["activity"],
        errors="coerce",
    )

    valid_label_mask = blind_activity.isin([0, 1]).to_numpy()

    # Keep labels in the output only after all probabilities exist.
    blind_output["true_activity"] = blind_activity.to_numpy()
    blind_output["true_class"] = np.where(
        blind_activity.to_numpy() == 1,
        "NADES",
        np.where(
            blind_activity.to_numpy() == 0,
            "non-NADES",
            pd.NA,
        ),
    )

    blind_output["correct_prediction"] = np.where(
        valid_label_mask,
        blind_prediction == blind_activity.to_numpy(),
        pd.NA,
    )

    if valid_label_mask.sum() > 0:
        y_blind = (
            blind_activity.to_numpy()[valid_label_mask]
            .astype(np.int8)
        )
        p_blind = blind_probability_nades[valid_label_mask]
        pred_blind = blind_prediction[valid_label_mask]

        n_non_nades = int(np.sum(y_blind == 0))
        n_nades = int(np.sum(y_blind == 1))

        blind_metric_values["blind_labeled_samples"] = int(len(y_blind))
        blind_metric_values["blind_n_NADES"] = n_nades
        blind_metric_values["blind_n_non_NADES"] = n_non_nades

        blind_metric_values["blind_balanced_accuracy"] = (
            balanced_accuracy_score(y_blind, pred_blind)
        )
        blind_metric_values["blind_mcc"] = matthews_corrcoef(
            y_blind,
            pred_blind,
        )
        blind_metric_values["blind_precision_NADES"] = precision_score(
            y_blind,
            pred_blind,
            zero_division=0,
        )
        blind_metric_values["blind_recall_NADES"] = recall_score(
            y_blind,
            pred_blind,
            zero_division=0,
        )
        blind_metric_values["blind_f1_NADES"] = f1_score(
            y_blind,
            pred_blind,
            zero_division=0,
        )
        blind_metric_values["blind_specificity_non_NADES"] = specificity(
            y_blind,
            pred_blind,
        )

        b_tn, b_fp, b_fn, b_tp = confusion_matrix(
            y_blind,
            pred_blind,
            labels=[0, 1],
        ).ravel()

        blind_metric_values.update(
            {
                "blind_tn": int(b_tn),
                "blind_fp": int(b_fp),
                "blind_fn": int(b_fn),
                "blind_tp": int(b_tp),
            }
        )

        if np.unique(y_blind).size == 2:
            blind_metric_values["blind_roc_auc"] = roc_auc_score(
                y_blind,
                p_blind,
            )
            blind_metric_values["blind_pr_auc_NADES"] = (
                average_precision_score(y_blind, p_blind)
            )
            blind_metric_values["blind_pr_auc_non_NADES"] = (
                average_precision_score(1 - y_blind, 1.0 - p_blind)
            )


# ============================================================
# 9. SAVE PREDICTIONS AND SUMMARY
# ============================================================

# Preserve the blind input order in the saved prediction table.
blind_output.to_csv(BLIND_PREDICTIONS_OUT, index=False)

n_predicted_nades = int(np.sum(blind_prediction == 1))
n_predicted_non_nades = int(np.sum(blind_prediction == 0))

summary_df = pd.DataFrame(
    [
        {
            "blind_samples": int(len(blind_output)),
            "selected_descriptor_count": len(selected_features),
            "threshold": THRESHOLD,
            "model_bundle": data_path('Final60_Internal_Evaluation/Final60_full_dataset_model_bundle.joblib'),
            "predicted_NADES": n_predicted_nades,
            "predicted_non_NADES": n_predicted_non_nades,
            "probability_NADES_min": float(blind_probability_nades.min()),
            "probability_NADES_median": float(
                np.median(blind_probability_nades)
            ),
            "probability_NADES_mean": float(
                blind_probability_nades.mean()
            ),
            "probability_NADES_max": float(blind_probability_nades.max()),
            **blind_metric_values,
        }
    ]
)

summary_df.to_csv(BLIND_SUMMARY_OUT, index=False)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 94)
print("FINAL60 BLIND-SET RESULTS")
print("=" * 94)
print(f"Blind samples                  : {len(blind_output)}")
print(f"Fixed descriptors              : {len(selected_features)}")
print(f"Frozen threshold               : {THRESHOLD:.2f}")
print(f"Predicted NADES                : {n_predicted_nades}")
print(f"Predicted non-NADES            : {n_predicted_non_nades}")
print(
    "NADES probability range       : "
    f"{blind_probability_nades.min():.4f}–"
    f"{blind_probability_nades.max():.4f}"
)

if blind_metric_values["blind_labeled_samples"] > 0:
    print("\nPost hoc external evaluation:")
    print(
        "Labeled blind samples          : "
        f"{blind_metric_values['blind_labeled_samples']}"
    )
    print(
        "NADES / non-NADES              : "
        f"{blind_metric_values['blind_n_NADES']}/"
        f"{blind_metric_values['blind_n_non_NADES']}"
    )

    print_metric("ROC-AUC", blind_metric_values["blind_roc_auc"])
    print_metric(
        "AP for NADES",
        blind_metric_values["blind_pr_auc_NADES"],
    )
    print_metric(
        "AP for non-NADES",
        blind_metric_values["blind_pr_auc_non_NADES"],
    )
    print_metric(
        "Balanced accuracy",
        blind_metric_values["blind_balanced_accuracy"],
    )
    print_metric("MCC", blind_metric_values["blind_mcc"])
    print_metric(
        "Precision, NADES",
        blind_metric_values["blind_precision_NADES"],
    )
    print_metric(
        "Recall, NADES",
        blind_metric_values["blind_recall_NADES"],
    )
    print_metric(
        "F1, NADES",
        blind_metric_values["blind_f1_NADES"],
    )
    print_metric(
        "Specificity, non-NADES",
        blind_metric_values["blind_specificity_non_NADES"],
    )

    print("\nBlind confusion matrix [[TN, FP], [FN, TP]]:")
    print(
        np.array(
            [
                [
                    blind_metric_values["blind_tn"],
                    blind_metric_values["blind_fp"],
                ],
                [
                    blind_metric_values["blind_fn"],
                    blind_metric_values["blind_tp"],
                ],
            ]
        )
    )

print("\nTen blind mixtures closest to the frozen threshold:")
print(
    blind_output
    .sort_values(["absolute_margin", "Code"])
    .head(10)
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}",
    )
)

print("\nSaved:")
print(" -", BLIND_PREDICTIONS_OUT)
print(" -", BLIND_SUMMARY_OUT)
print("\nNo blind labels were used to alter the model or threshold.")

