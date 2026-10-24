
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


"""
FINAL60 — TRAIN-ONLY STRATIFIED 5-FOLD OOF ROC/PR CURVES
=========================================================

Run this cell/script only AFTER Final60_Internal_Evaluation.py.  It uses the
already defined Final60 training variables:

    X_train_raw, y_train, codes_train, XGB_PARAMS, THRESHOLD, OUTPUT_DIR

It deliberately uses only the 511-row training partition.  The fixed
128-sample Internal_evaluation set and the blind set are not read or used.

The threshold is inherited unchanged from the prior Final60 pipeline:
    TAU = float(THRESHOLD) = 0.42
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. SETTINGS
# ============================================================

NFOLDS = 5
SPLIT_SEED = 42
SELECTED_SUBSET_SIZE = 60
EXPECTED_FEATURE_COUNT = 60
TAU = float(THRESHOLD)  # frozen Final60 threshold; do not edit here

plt.rcParams["font.family"] = "Arial"

FIG_W, FIG_H = 12, 6
FS, TICK_FS, LEGEND_FS = 14, 12, 12
LW_MEAN, LW_FOLD, LW_REFERENCE = 2.0, 1.0, 1.0
ALPHA_FOLD, ALPHA_VARIATION = 0.25, 0.20

CURVE_COLOR = "#00A087FF"
REFERENCE_COLOR = "#999999"

ROC_GRID = np.linspace(0.0, 1.0, 500)
PR_GRID = np.linspace(0.0, 1.0, 500)

CV_ROC_PR_PNG = OUTPUT_DIR / "Final60_TrainOnly_CV_ROC_PR.png"
CV_ROC_PR_PDF = OUTPUT_DIR / "Final60_TrainOnly_CV_ROC_PR.pdf"
CV_OOF_OUT = OUTPUT_DIR / "Final60_TrainOnly_CV_OOF_predictions.csv"
CV_FOLD_METRICS_OUT = OUTPUT_DIR / "Final60_TrainOnly_CV_fold_metrics.csv"
CV_SUMMARY_OUT = OUTPUT_DIR / "Final60_TrainOnly_CV_summary_metrics.csv"


# ============================================================
# 2. VALIDATE FINAL60 TRAINING INPUTS
# ============================================================

required_variables = [
    "X_train_raw",
    "y_train",
    "codes_train",
    "XGB_PARAMS",
    "THRESHOLD",
    "OUTPUT_DIR",
]
missing_variables = [name for name in required_variables if name not in globals()]
if missing_variables:
    raise NameError(
        "Run Final60_Internal_Evaluation.py first. Missing variables: "
        f"{missing_variables}"
    )

if not isinstance(X_train_raw, pd.DataFrame):
    raise TypeError("X_train_raw must be a pandas DataFrame.")

y_cv = np.asarray(y_train, dtype=np.int8)
cv_codes = pd.Series(codes_train).astype("string").reset_index(drop=True)
X_cv_raw = X_train_raw.reset_index(drop=True).copy()

if len(X_cv_raw) != len(y_cv) or len(cv_codes) != len(y_cv):
    raise ValueError("X_train_raw, y_train, and codes_train must have the same length.")
if X_cv_raw.shape[1] != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected Final60 input with {EXPECTED_FEATURE_COUNT} descriptors; "
        f"found {X_cv_raw.shape[1]}."
    )
if set(np.unique(y_cv)) != {0, 1}:
    raise ValueError("The training partition must contain both classes 0 and 1.")
if not np.isclose(TAU, 0.42):
    print(f"WARNING: inherited THRESHOLD is {TAU:.2f}, not the expected Final60 value 0.42.")

print("=" * 94)
print("FINAL60 TRAIN-ONLY STRATIFIED 5-FOLD OOF")
print("=" * 94)
print("Samples used              :", len(y_cv))
print("Class balance             :", dict(pd.Series(y_cv).value_counts().sort_index()))
print("Fixed threshold           :", f"{TAU:.2f}")
print("Internal evaluation used? : NO")
print("Blind data used?          : NO")


# ============================================================
# 3. HELPERS
# ============================================================

def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "specificity_non_NADES": float(specificity),
        "precision_NADES": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_NADES": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_NADES": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def interpolate_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """Interpolate one PR curve on ascending recall with stable duplicates."""
    recall_ascending = recall[::-1]
    precision_ascending = precision[::-1]
    recall_unique, unique_index = np.unique(recall_ascending, return_index=True)
    precision_unique = precision_ascending[unique_index]
    return np.interp(PR_GRID, recall_unique, precision_unique)


# ============================================================
# 4. FIVE-FOLD OOF PREDICTIONS
# ============================================================

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SPLIT_SEED)

oof_probability = np.full(len(y_cv), np.nan, dtype=float)
oof_fold = np.full(len(y_cv), -1, dtype=int)

roc_aucs = []
pr_aps = []
tprs_interp = []
precs_interp = []
fold_records = []

for fold, (fit_idx, valid_idx) in enumerate(skf.split(X_cv_raw, y_cv), start=1):
    X_fit_raw = X_cv_raw.iloc[fit_idx]
    X_valid_raw = X_cv_raw.iloc[valid_idx]
    y_fit = y_cv[fit_idx]
    y_valid = y_cv[valid_idx]

    # All preprocessing is fitted only on the fold-training rows.
    imputer = SimpleImputer(strategy="median")
    X_fit = imputer.fit_transform(X_fit_raw).astype(np.float32)
    X_valid = imputer.transform(X_valid_raw).astype(np.float32)

    n_non_nades = int(np.sum(y_fit == 0))
    n_nades = int(np.sum(y_fit == 1))
    if n_non_nades == 0 or n_nades == 0:
        raise ValueError(f"Fold {fold} training data do not contain both classes.")

    # Exact seed convention used by the clean DAGA subset-size sweep for k=60.
    fold_seed = SPLIT_SEED + SELECTED_SUBSET_SIZE * 100 + fold
    fold_params = dict(XGB_PARAMS)
    fold_params["random_state"] = fold_seed

    model = xgb.XGBClassifier(
        **fold_params,
        scale_pos_weight=n_non_nades / n_nades,
    )
    model.fit(X_fit, y_fit)
    probability = model.predict_proba(X_valid)[:, 1]

    oof_probability[valid_idx] = probability
    oof_fold[valid_idx] = fold

    fold_fpr, fold_tpr, _ = roc_curve(y_valid, probability)
    fold_precision, fold_recall, _ = precision_recall_curve(y_valid, probability)

    fold_roc_auc = float(roc_auc_score(y_valid, probability))
    fold_ap = float(average_precision_score(y_valid, probability))
    prediction_metrics = threshold_metrics(y_valid, probability, TAU)

    roc_aucs.append(fold_roc_auc)
    pr_aps.append(fold_ap)

    tpr_interp = np.interp(ROC_GRID, fold_fpr, fold_tpr)
    tpr_interp[0] = 0.0
    tpr_interp[-1] = 1.0
    tprs_interp.append(tpr_interp)
    precs_interp.append(interpolate_pr_curve(fold_recall, fold_precision))

    fold_records.append({
        "fold": fold,
        "model_seed": fold_seed,
        "n_train": int(len(fit_idx)),
        "n_validation": int(len(valid_idx)),
        "n_train_non_NADES": n_non_nades,
        "n_train_NADES": n_nades,
        "scale_pos_weight": float(n_non_nades / n_nades),
        "threshold": TAU,
        "roc_auc": fold_roc_auc,
        "average_precision_NADES": fold_ap,
        **prediction_metrics,
    })

    print(
        f"Fold {fold}/{NFOLDS}: ROC-AUC={fold_roc_auc:.4f} | "
        f"AP={fold_ap:.4f} | BA={prediction_metrics['balanced_accuracy']:.4f} | "
        f"MCC={prediction_metrics['mcc']:.4f} | "
        f"Specificity={prediction_metrics['specificity_non_NADES']:.4f} | "
        f"Recall={prediction_metrics['recall_NADES']:.4f}"
    )

if np.isnan(oof_probability).any() or np.any(oof_fold < 1):
    raise RuntimeError("Some training samples did not receive OOF predictions.")


# ============================================================
# 5. COMBINED OOF METRICS AND CSV OUTPUTS
# ============================================================

oof_prediction = (oof_probability >= TAU).astype(np.int8)
oof_metrics = {
    "evaluation": "Train-only combined OOF",
    "n_samples": int(len(y_cv)),
    "n_folds": NFOLDS,
    "threshold": TAU,
    "roc_auc": float(roc_auc_score(y_cv, oof_probability)),
    "average_precision_NADES": float(average_precision_score(y_cv, oof_probability)),
    "average_precision_non_NADES": float(
        average_precision_score(1 - y_cv, 1.0 - oof_probability)
    ),
    **threshold_metrics(y_cv, oof_probability, TAU),
}

fold_metrics = pd.DataFrame(fold_records)
fold_metrics.to_csv(CV_FOLD_METRICS_OUT, index=False)

summary_metrics = [
    "roc_auc", "average_precision_NADES", "accuracy", "balanced_accuracy",
    "mcc", "specificity_non_NADES", "precision_NADES", "recall_NADES", "f1_NADES",
]
summary_table = pd.DataFrame([
    {
        "metric": metric,
        "fold_mean": float(fold_metrics[metric].mean()),
        "fold_std": float(fold_metrics[metric].std(ddof=1)),
        "fold_min": float(fold_metrics[metric].min()),
        "fold_max": float(fold_metrics[metric].max()),
        "combined_oof_value": float(oof_metrics[metric]),
    }
    for metric in summary_metrics
])
summary_table.to_csv(CV_SUMMARY_OUT, index=False)

oof_output = pd.DataFrame({
    "Code": cv_codes,
    "activity": y_cv,
    "true_class": np.where(y_cv == 1, "NADES", "non-NADES"),
    "fold": oof_fold,
    "probability_NADES": oof_probability,
    "probability_non_NADES": 1.0 - oof_probability,
    "threshold_used": TAU,
    "predicted_activity": oof_prediction,
    "predicted_class": np.where(oof_prediction == 1, "NADES", "non-NADES"),
    "correct_prediction": oof_prediction == y_cv,
    "signed_margin": oof_probability - TAU,
    "absolute_margin": np.abs(oof_probability - TAU),
})
oof_output.to_csv(CV_OOF_OUT, index=False)


# ============================================================
# 6. MEAN FOLD ROC/PR CURVES
# ============================================================

tprs_array = np.asarray(tprs_interp)
precs_array = np.asarray(precs_interp)

mean_tpr = tprs_array.mean(axis=0)
mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
std_tpr = tprs_array.std(axis=0, ddof=1)

mean_precision = precs_array.mean(axis=0)
std_precision = precs_array.std(axis=0, ddof=1)

mean_roc_auc = float(np.mean(roc_aucs))
std_roc_auc = float(np.std(roc_aucs, ddof=1))
mean_ap = float(np.mean(pr_aps))
std_ap = float(np.std(pr_aps, ddof=1))
positive_prevalence = float(np.mean(y_cv == 1))


# ============================================================
# 7. PLOT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

for fold_number, fold_tpr in enumerate(tprs_array, start=1):
    axes[0].plot(
        ROC_GRID, fold_tpr,
        alpha=ALPHA_FOLD,
        linewidth=LW_FOLD,
        color=CURVE_COLOR,
        label="Individual folds" if fold_number == 1 else None,
    )

axes[0].plot(
    ROC_GRID, mean_tpr,
    linewidth=LW_MEAN,
    color=CURVE_COLOR,
    label=f"Mean ROC (AUC = {mean_roc_auc:.3f} ± {std_roc_auc:.3f})",
)
axes[0].fill_between(
    ROC_GRID,
    np.clip(mean_tpr - std_tpr, 0.0, 1.0),
    np.clip(mean_tpr + std_tpr, 0.0, 1.0),
    color=CURVE_COLOR,
    alpha=ALPHA_VARIATION,
    label="Fold-to-fold SD",
)
axes[0].plot([0, 1], [0, 1], "--", linewidth=LW_REFERENCE, color=REFERENCE_COLOR)
axes[0].set_xlabel("False Positive Rate", fontsize=FS)
axes[0].set_ylabel("True Positive Rate", fontsize=FS)
axes[0].set_xlim(0.0, 1.0)
axes[0].set_ylim(0.0, 1.02)
axes[0].tick_params(axis="both", labelsize=TICK_FS)
axes[0].legend(loc="lower right", frameon=False, fontsize=LEGEND_FS)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].grid(False)
axes[0].text(-0.14, 1.03, "", transform=axes[0].transAxes, fontsize=16, fontweight="bold", va="top")

for fold_number, fold_precision in enumerate(precs_array, start=1):
    axes[1].plot(
        PR_GRID, fold_precision,
        alpha=ALPHA_FOLD,
        linewidth=LW_FOLD,
        color=CURVE_COLOR,
        label="Individual folds" if fold_number == 1 else None,
    )

axes[1].plot(
    PR_GRID, mean_precision,
    linewidth=LW_MEAN,
    color=CURVE_COLOR,
    label=f"Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})",
)
axes[1].fill_between(
    PR_GRID,
    np.clip(mean_precision - std_precision, 0.0, 1.0),
    np.clip(mean_precision + std_precision, 0.0, 1.0),
    color=CURVE_COLOR,
    alpha=ALPHA_VARIATION,
    label="Fold-to-fold SD",
)
axes[1].axhline(
    positive_prevalence,
    linestyle="--",
    linewidth=LW_REFERENCE,
    color=REFERENCE_COLOR,
    label=f"Baseline = {positive_prevalence:.3f}",
)
axes[1].set_xlabel("Recall", fontsize=FS)
axes[1].set_ylabel("Precision", fontsize=FS)
axes[1].set_xlim(0.0, 1.0)
axes[1].set_ylim(0.0, 1.02)
axes[1].tick_params(axis="both", labelsize=TICK_FS)
axes[1].legend(loc="lower left", frameon=False, fontsize=LEGEND_FS)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].grid(False)
axes[1].text(-0.14, 1.03, "", transform=axes[1].transAxes, fontsize=16, fontweight="bold", va="top")

fig.tight_layout()
fig.savefig(CV_ROC_PR_PNG, dpi=600, bbox_inches="tight")
fig.savefig(CV_ROC_PR_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 8. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 94)
print("FINAL60 TRAIN-ONLY STRATIFIED 5-FOLD OOF RESULTS")
print("=" * 94)
print(f"Mean fold ROC-AUC          : {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")
print(f"Mean fold AP (NADES)       : {mean_ap:.4f} ± {std_ap:.4f}")
print("\nCombined OOF results:")
print(f"OOF ROC-AUC                : {oof_metrics['roc_auc']:.4f}")
print(f"OOF AP NADES               : {oof_metrics['average_precision_NADES']:.4f}")
print(f"OOF AP non-NADES           : {oof_metrics['average_precision_non_NADES']:.4f}")
print(f"OOF fixed threshold         : {TAU:.2f}")
print(f"OOF balanced accuracy       : {oof_metrics['balanced_accuracy']:.4f}")
print(f"OOF MCC                    : {oof_metrics['mcc']:.4f}")
print(f"OOF specificity non-NADES  : {oof_metrics['specificity_non_NADES']:.4f}")
print(f"OOF recall NADES           : {oof_metrics['recall_NADES']:.4f}")
print("\nOOF confusion matrix [[TN, FP], [FN, TP]]:")
print(np.array([[oof_metrics["tn"], oof_metrics["fp"]], [oof_metrics["fn"], oof_metrics["tp"]]]))
print("\nSaved:")
for path in [CV_ROC_PR_PNG, CV_ROC_PR_PDF, CV_OOF_OUT, CV_FOLD_METRICS_OUT, CV_SUMMARY_OUT]:
    print(" -", path)
print("\nInternal-evaluation and blind data were not used.")


