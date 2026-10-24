
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
# FINAL60 COMPONENT_B-ONLY — TRAIN-ONLY 5-FOLD OOF TEST
# ============================================================
#
# Scientific purpose
# ------------------
# The blind-family diagnostic suggested that the Component_B descriptor
# family may carry strong class-separation information. This script tests
# that hypothesis WITHOUT loading the 128 internal-evaluation samples or
# the blind set.
#
# Protocol
# --------
# 1. Reuse the fixed seed-42 split.
# 2. Keep only the 511-row Train partition.
# 3. Obtain Component_B descriptors from the already fixed Final60 list.
# 4. In every OOF fold, fit the imputer and XGBoost model on that fold's
#    inner-training data only; predict only its validation fold.
# 5. Evaluate combined OOF probabilities using the pre-existing Final60
#    operating threshold tau = 0.42.
#
# This script intentionally does NOT fit a deployment model, access blind
# labels, alter Final60, select a new threshold, or use the held-out 128.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
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
# 1. PATHS AND FIXED SETTINGS
# ============================================================

BASE = data_path('')

TRAIN_PATH = data_path('master_train_features.csv')
SPLIT_PATH = data_path('Final_clean_split_seed42.csv')
CANDIDATE_PATH = data_path('GA_candidate_descriptors_1000.csv')
SELECTED_PATH = (
    data_path('Final60_Internal_Evaluation/Final60_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('Final60_Component_B_TrainOnly_OOF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOF_PREDICTIONS_OUT = (
    data_path('Final60_Component_B_TrainOnly_OOF/Final60_Component_B_train_only_OOF_predictions.csv')
)
FOLD_METRICS_OUT = (
    data_path('Final60_Component_B_TrainOnly_OOF/Final60_Component_B_train_only_fold_metrics.csv')
)
SUMMARY_OUT = (
    data_path('Final60_Component_B_TrainOnly_OOF/Final60_Component_B_train_only_OOF_summary.csv')
)
ROC_PR_PNG = (
    data_path('Final60_Component_B_TrainOnly_OOF/Final60_Component_B_train_only_OOF_ROC_PR.png')
)
ROC_PR_PDF = (
    data_path('Final60_Component_B_TrainOnly_OOF/Final60_Component_B_train_only_OOF_ROC_PR.pdf')
)

TARGET_COLUMN = "activity"
CODE_COLUMN = "Code"
FAMILY_TO_TEST = "Component_B"

SPLIT_TRAIN_LABEL = "Train"
SEED = 42
N_SPLITS = 5

# This is the threshold frozen for the complete Final60 model.
# It is reused only as an operating-point reference here.
FROZEN_THRESHOLD = 0.42

EXPECTED_FEATURE_COUNT = 60
MODEL_SEED = 82

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

def find_column(frame, candidates, file_label):
    """Find a column name case-insensitively."""
    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find one of {candidates} in {file_label}.\n"
        f"Available columns: {frame.columns.tolist()}"
    )


def specificity_from_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return tn, fp, fn, tp, specificity


def build_model(scale_pos_weight):
    return xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )


# ============================================================
# 3. READ THE FIXED DESCRIPTOR FAMILY
# ============================================================

for path, description in [
    (TRAIN_PATH, "Labelled feature file"),
    (SPLIT_PATH, "Fixed clean split"),
    (CANDIDATE_PATH, "GA candidate descriptor file"),
    (SELECTED_PATH, "Final60 selected descriptor file"),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )

selected_table = pd.read_csv(SELECTED_PATH, low_memory=False)
candidate_table = pd.read_csv(CANDIDATE_PATH, low_memory=False)

feature_candidates = [
    "feature",
    "descriptor",
    "descriptor_name",
    "selected_feature",
]
family_candidates = [
    "family",
    "descriptor_family",
    "feature_family",
]

selected_feature_col = find_column(
    selected_table,
    feature_candidates,
    SELECTED_PATH.name,
)
selected_family_col = find_column(
    selected_table,
    family_candidates,
    SELECTED_PATH.name,
)

candidate_feature_col = find_column(
    candidate_table,
    feature_candidates,
    CANDIDATE_PATH.name,
)
candidate_family_col = find_column(
    candidate_table,
    family_candidates,
    CANDIDATE_PATH.name,
)

selected_metadata = (
    selected_table[[selected_feature_col, selected_family_col]]
    .rename(
        columns={
            selected_feature_col: "feature",
            selected_family_col: "family_selected",
        }
    )
    .dropna()
    .copy()
)

candidate_metadata = (
    candidate_table[[candidate_feature_col, candidate_family_col]]
    .rename(
        columns={
            candidate_feature_col: "feature",
            candidate_family_col: "family_candidate",
        }
    )
    .dropna()
    .drop_duplicates(subset="feature", keep="first")
    .copy()
)

for column in ["feature", "family_selected"]:
    selected_metadata[column] = (
        selected_metadata[column].astype(str).str.strip()
    )

for column in ["feature", "family_candidate"]:
    candidate_metadata[column] = (
        candidate_metadata[column].astype(str).str.strip()
    )

if selected_metadata["feature"].duplicated().any():
    raise ValueError(
        "Final60_selected_descriptors.csv has duplicated features."
    )

metadata = selected_metadata.merge(
    candidate_metadata,
    on="feature",
    how="left",
    validate="one_to_one",
)

if metadata["family_candidate"].isna().any():
    missing = metadata.loc[
        metadata["family_candidate"].isna(),
        "feature",
    ].tolist()
    raise ValueError(
        "Final60 descriptor(s) missing from candidate pool:\n"
        + "\n".join(missing)
    )

disagreement = metadata.loc[
    metadata["family_selected"] != metadata["family_candidate"],
    ["feature", "family_selected", "family_candidate"],
]

if not disagreement.empty:
    raise ValueError(
        "Family labels disagree between Final60_selected_descriptors.csv "
        "and GA_candidate_descriptors_1000.csv:\n"
        + disagreement.to_string(index=False)
    )

if len(metadata) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} Final60 descriptors, "
        f"found {len(metadata)}."
    )

family_features = metadata.loc[
    metadata["family_selected"] == FAMILY_TO_TEST,
    "feature",
].tolist()

if not family_features:
    available_families = sorted(
        metadata["family_selected"].unique().tolist()
    )
    raise ValueError(
        f"Family '{FAMILY_TO_TEST}' is absent from Final60.\n"
        f"Available families: {available_families}"
    )


# ============================================================
# 4. LOAD ONLY THE FIXED TRAIN PARTITION (n = 511)
# ============================================================

data = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

if not {CODE_COLUMN, TARGET_COLUMN}.issubset(data.columns):
    raise ValueError(
        f"{TRAIN_PATH.name} must contain '{CODE_COLUMN}' and "
        f"'{TARGET_COLUMN}'."
    )

y_numeric = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
valid_labels = y_numeric.isin([0, 1])
data = data.loc[valid_labels].copy()
y_numeric = y_numeric.loc[valid_labels].astype(np.int8)

data[CODE_COLUMN] = data[CODE_COLUMN].astype("string").str.strip()

if data[CODE_COLUMN].isna().any() or data[CODE_COLUMN].duplicated().any():
    raise ValueError("Labelled Code values must be complete and unique.")

missing_features = [
    feature for feature in family_features
    if feature not in data.columns
]

if missing_features:
    raise ValueError(
        "Component_B feature(s) missing from labelled data:\n"
        + "\n".join(missing_features)
    )

split_ref = pd.read_csv(SPLIT_PATH, low_memory=False)

if not {CODE_COLUMN, "split"}.issubset(split_ref.columns):
    raise ValueError(
        f"{SPLIT_PATH.name} must contain 'Code' and 'split'."
    )

split_ref[CODE_COLUMN] = (
    split_ref[CODE_COLUMN].astype("string").str.strip()
)

split_map = dict(
    zip(split_ref[CODE_COLUMN], split_ref["split"])
)

split_label = data[CODE_COLUMN].map(split_map)

if split_label.isna().any():
    examples = data.loc[
        split_label.isna(),
        CODE_COLUMN,
    ].head(20).tolist()
    raise ValueError(
        "Some labelled Codes are missing from the fixed split. "
        f"Examples: {examples}"
    )

is_train = (split_label == SPLIT_TRAIN_LABEL).to_numpy()

if is_train.sum() != 511:
    raise ValueError(
        f"Expected 511 fixed Train samples, found {is_train.sum()}.\n"
        f"Observed split labels: {sorted(pd.unique(split_label).tolist())}"
    )

X_train_raw = (
    data.loc[is_train, family_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .reset_index(drop=True)
)

y_train = y_numeric.loc[is_train].to_numpy(dtype=np.int8)
codes_train = (
    data.loc[is_train, CODE_COLUMN]
    .reset_index(drop=True)
    .astype(str)
)

if len(np.unique(y_train)) != 2:
    raise ValueError("Both classes must occur in the fixed train partition.")


# ============================================================
# 5. STRICT TRAIN-ONLY STRATIFIED OOF PREDICTIONS
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

oof_probability = np.full(len(y_train), np.nan, dtype=float)
fold_id = np.full(len(y_train), -1, dtype=int)
fold_rows = []

print("=" * 94)
print("FINAL60 COMPONENT_B-ONLY — TRAIN-ONLY STRATIFIED 5-FOLD OOF")
print("=" * 94)
print(f"Fixed train samples        : {len(y_train)}")
print(
    "NADES / non-NADES         : "
    f"{int((y_train == 1).sum())} / {int((y_train == 0).sum())}"
)
print(f"Component_B descriptors    : {len(family_features)}")
print(f"Descriptors                : {family_features}")
print(f"Operating threshold         : {FROZEN_THRESHOLD:.2f}")
print("Internal hold-out and blind data are not loaded.\n")

for fold, (inner_train_idx, validation_idx) in enumerate(
    skf.split(X_train_raw, y_train),
    start=1,
):
    X_inner_train_raw = X_train_raw.iloc[inner_train_idx]
    X_validation_raw = X_train_raw.iloc[validation_idx]
    y_inner_train = y_train[inner_train_idx]
    y_validation = y_train[validation_idx]

    # Strictly fold-local preprocessing.
    imputer = SimpleImputer(strategy="median")
    X_inner_train = imputer.fit_transform(
        X_inner_train_raw
    ).astype(np.float32)
    X_validation = imputer.transform(
        X_validation_raw
    ).astype(np.float32)

    n_pos_inner = int((y_inner_train == 1).sum())
    n_neg_inner = int((y_inner_train == 0).sum())
    inner_spw = n_neg_inner / n_pos_inner

    model = build_model(inner_spw)
    model.fit(X_inner_train, y_inner_train)

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    oof_probability[validation_idx] = validation_probability
    fold_id[validation_idx] = fold

    y_validation_pred = (
        validation_probability >= FROZEN_THRESHOLD
    ).astype(np.int8)

    _, _, _, _, fold_specificity = specificity_from_confusion(
        y_validation,
        y_validation_pred,
    )

    fold_rows.append(
        {
            "fold": fold,
            "n_validation": len(validation_idx),
            "roc_auc": roc_auc_score(
                y_validation,
                validation_probability,
            ),
            "ap_nades": average_precision_score(
                y_validation,
                validation_probability,
            ),
            "balanced_accuracy_at_tau": balanced_accuracy_score(
                y_validation,
                y_validation_pred,
            ),
            "mcc_at_tau": matthews_corrcoef(
                y_validation,
                y_validation_pred,
            ),
            "specificity_non_nades_at_tau": fold_specificity,
            "recall_nades_at_tau": recall_score(
                y_validation,
                y_validation_pred,
                zero_division=0,
            ),
        }
    )

    print(
        f"Fold {fold}/{N_SPLITS} completed | "
        f"validation n = {len(validation_idx)}"
    )

if np.isnan(oof_probability).any() or (fold_id < 1).any():
    raise RuntimeError("OOF predictions are incomplete.")


# ============================================================
# 6. COMBINED OOF METRICS AT THE FROZEN THRESHOLD
# ============================================================

oof_prediction = (
    oof_probability >= FROZEN_THRESHOLD
).astype(np.int8)

tn, fp, fn, tp, specificity = specificity_from_confusion(
    y_train,
    oof_prediction,
)

metrics = {
    "family": FAMILY_TO_TEST,
    "n_features": len(family_features),
    "n_train_samples": len(y_train),
    "n_nades": int((y_train == 1).sum()),
    "n_non_nades": int((y_train == 0).sum()),
    "threshold": FROZEN_THRESHOLD,
    "roc_auc": roc_auc_score(y_train, oof_probability),
    "ap_nades": average_precision_score(y_train, oof_probability),
    "ap_non_nades": average_precision_score(
        1 - y_train,
        1 - oof_probability,
    ),
    "brier_score": brier_score_loss(y_train, oof_probability),
    "balanced_accuracy": balanced_accuracy_score(
        y_train,
        oof_prediction,
    ),
    "mcc": matthews_corrcoef(y_train, oof_prediction),
    "precision_nades": precision_score(
        y_train,
        oof_prediction,
        zero_division=0,
    ),
    "recall_nades": recall_score(
        y_train,
        oof_prediction,
        zero_division=0,
    ),
    "f1_nades": f1_score(
        y_train,
        oof_prediction,
        zero_division=0,
    ),
    "specificity_non_nades": specificity,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
}

fold_metrics = pd.DataFrame(fold_rows)

oof_output = pd.DataFrame(
    {
        "Code": codes_train,
        "fold": fold_id,
        "activity": y_train,
        "true_class": np.where(
            y_train == 1,
            "NADES",
            "non-NADES",
        ),
        "probability_NADES": oof_probability,
        "threshold_used": FROZEN_THRESHOLD,
        "predicted_activity": oof_prediction,
        "predicted_class": np.where(
            oof_prediction == 1,
            "NADES",
            "non-NADES",
        ),
        "correct_prediction": oof_prediction == y_train,
    }
)

oof_output.to_csv(OOF_PREDICTIONS_OUT, index=False)
fold_metrics.to_csv(FOLD_METRICS_OUT, index=False)
pd.DataFrame([metrics]).to_csv(SUMMARY_OUT, index=False)


# ============================================================
# 7. OOF ROC AND PR CURVES
# ============================================================

fpr, tpr, _ = roc_curve(y_train, oof_probability)
precision_curve, recall_curve, _ = precision_recall_curve(
    y_train,
    oof_probability,
)

positive_prevalence = float(np.mean(y_train == 1))
precision_at_tau = metrics["precision_nades"]
recall_at_tau = metrics["recall_nades"]
fpr_at_tau = fp / (fp + tn)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

axes[0].plot(
    fpr,
    tpr,
    color="#298C8C",
    linewidth=2.2,
    label=f"Component_B OOF (AUC = {metrics['roc_auc']:.3f})",
)
axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.2,
    color="#999999",
)
axes[0].scatter(
    fpr_at_tau,
    recall_at_tau,
    s=65,
    color="#2A6FBB",
    edgecolor="white",
    linewidth=0.7,
    zorder=5,
)
axes[0].annotate(
    rf"$\tau = {FROZEN_THRESHOLD:.2f}$",
    xy=(fpr_at_tau, recall_at_tau),
    xytext=(6, -8),
    textcoords="offset points",
    fontsize=11,
    color="#2A6FBB",
)
axes[0].set_xlabel("False Positive Rate", fontsize=14)
axes[0].set_ylabel("True Positive Rate", fontsize=14)
axes[0].set_xlim(0.0, 1.0)
axes[0].set_ylim(0.0, 1.02)
axes[0].tick_params(axis="both", labelsize=12)
axes[0].legend(frameon=False, fontsize=11, loc="lower right")

axes[1].plot(
    recall_curve,
    precision_curve,
    color="#298C8C",
    linewidth=2.2,
    label=f"Component_B OOF (AP = {metrics['ap_nades']:.3f})",
)
axes[1].axhline(
    positive_prevalence,
    linestyle="--",
    linewidth=1.2,
    color="#999999",
    label=f"NADES prevalence = {positive_prevalence:.3f}",
)
axes[1].scatter(
    recall_at_tau,
    precision_at_tau,
    s=65,
    color="#2A6FBB",
    edgecolor="white",
    linewidth=0.7,
    zorder=5,
)
axes[1].annotate(
    rf"$\tau = {FROZEN_THRESHOLD:.2f}$",
    xy=(recall_at_tau, precision_at_tau),
    xytext=(6, -8),
    textcoords="offset points",
    fontsize=11,
    color="#2A6FBB",
)
axes[1].set_xlabel("Recall", fontsize=14)
axes[1].set_ylabel("Precision", fontsize=14)
axes[1].set_xlim(0.0, 1.0)
axes[1].set_ylim(0.0, 1.02)
axes[1].tick_params(axis="both", labelsize=12)
axes[1].legend(frameon=False, fontsize=11, loc="lower left")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

axes[0].text(
    -0.14,
    1.03,
    "A",
    transform=axes[0].transAxes,
    fontsize=16,
    fontweight="bold",
)
axes[1].text(
    -0.14,
    1.03,
    "B",
    transform=axes[1].transAxes,
    fontsize=16,
    fontweight="bold",
)

fig.tight_layout()
fig.savefig(ROC_PR_PNG, dpi=600, bbox_inches="tight")
fig.savefig(ROC_PR_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 8. REPORT
# ============================================================

print("\n" + "=" * 94)
print("COMPONENT_B-ONLY TRAIN-ONLY OOF RESULTS")
print("=" * 94)
print(f"OOF ROC-AUC                : {metrics['roc_auc']:.4f}")
print(f"OOF AP for NADES           : {metrics['ap_nades']:.4f}")
print(f"OOF AP for non-NADES       : {metrics['ap_non_nades']:.4f}")
print(f"OOF Brier score            : {metrics['brier_score']:.4f}")
print(f"OOF fixed threshold        : {FROZEN_THRESHOLD:.2f}")
print(f"OOF balanced accuracy      : {metrics['balanced_accuracy']:.4f}")
print(f"OOF MCC                    : {metrics['mcc']:.4f}")
print(f"OOF specificity non-NADES  : {metrics['specificity_non_nades']:.4f}")
print(f"OOF recall NADES           : {metrics['recall_nades']:.4f}")
print("\nOOF confusion matrix [[TN, FP], [FN, TP]]:")
print(np.array([[tn, fp], [fn, tp]]))

print("\nMean fold metrics ± SD:")
for metric_name in ["roc_auc", "ap_nades"]:
    print(
        f"{metric_name:28s}: "
        f"{fold_metrics[metric_name].mean():.4f} "
        f"± {fold_metrics[metric_name].std(ddof=1):.4f}"
    )

print("\nSaved:")
for path in [
    OOF_PREDICTIONS_OUT,
    FOLD_METRICS_OUT,
    SUMMARY_OUT,
    ROC_PR_PNG,
    ROC_PR_PDF,
]:
    print(" -", path)

print(
    "\nInterpretation: this is a train-only OOF hypothesis check. "
    "The blind set remains untouched by this script."
)

