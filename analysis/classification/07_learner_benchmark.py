

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
TRAIN-ONLY OOF MODEL-FAMILY BENCHMARK
======================================

Compares five fixed model families on the SAME clean 60-descriptor subset:

    1. Logistic Regression       — linear baseline
    2. Random Forest             — bagged-tree baseline
    3. Extra Trees               — randomized-tree baseline
    4. SVM-RBF                   — nonlinear non-tree baseline
    5. XGBoost                   — final model family

Protocol:
    * Uses only rows labelled split == 'Train' in Final_clean_split_seed42.csv.
    * Each model receives identical five-fold stratified OOF folds.
    * Every imputer/scaler/model is fitted inside the corresponding fold.
    * Threshold-independent metrics use OOF probabilities.
    * Each operating threshold is selected from OOF balanced accuracy, with
      NADES recall used only as a deterministic tie-breaker.

The 128 fixed Internal_evaluation samples and the blind set are NOT loaded or
used.  This is a development benchmark on an XGBoost-guided descriptor subset,
not an independently tuned contest between model families.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 0. PATHS
# ============================================================

BASE = data_path('')

DATA_PATH = data_path('master_train_features.csv')
SPLIT_PATH = data_path('Final_clean_split_seed42.csv')

# Use the 60-descriptor representation selected by the clean, train-only DAGA sweep.
SELECTED_SUBSET_SIZE = 60
SELECTED_FEATURES_PATH = (
    data_path('DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('Final60_Model_Family_Benchmark_OOF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_OUT = data_path('Final60_Model_Family_Benchmark_OOF/model_family_OOF_metrics.csv')
PREDICTIONS_OUT = data_path('Final60_Model_Family_Benchmark_OOF/model_family_OOF_predictions_long.csv')
SETTINGS_OUT = data_path('Final60_Model_Family_Benchmark_OOF/model_family_benchmark_settings.csv')
ROC_PR_PNG = data_path('Final60_Model_Family_Benchmark_OOF/model_family_OOF_ROC_PR.png')
ROC_PR_PDF = data_path('Final60_Model_Family_Benchmark_OOF/model_family_OOF_ROC_PR.pdf')
RELIABILITY_PNG = data_path('Final60_Model_Family_Benchmark_OOF/model_family_OOF_reliability.png')
RELIABILITY_PDF = data_path('Final60_Model_Family_Benchmark_OOF/model_family_OOF_reliability.pdf')


# ============================================================
# 1. FIXED SETTINGS
# ============================================================

SEED = 42
MODEL_SEED = 82
N_SPLITS = 5

TARGET_COLUMN = "activity"
CODE_COLUMN = "Code"
SPLIT_COLUMN = "split"
TRAIN_SPLIT_LABEL = "Train"

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

# Thresholds are an operating decision, separate from ROC-AUC and AP.
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)

# All fixed settings are deliberately modest baseline settings, not a new
# hyperparameter search.  This avoids tuning one comparator more than another.
LOGISTIC_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 5000,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}

EXTRA_TREES_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "n_jobs": -1,
}

SVM_RBF_PARAMS = {
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

MODEL_ORDER = [
    "Logistic regression",
    "Random forest",
    "Extra trees",
    "SVM-RBF",
    "XGBoost",
]

MODEL_COLORS = {
    "Logistic regression": "#4C78A8",
    "Random forest": "#8C6BB1",
    "Extra trees": "#54A24B",
    "SVM-RBF": "#E17C05",
    "XGBoost": "#298C8C",
}

plt.rcParams["font.family"] = "Arial"


# ============================================================
# 2. HELPERS
# ============================================================

def validate_binary_target(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError("activity contains missing or non-numeric values.")
    unexpected = set(values.unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"activity must contain only 0 and 1, found {unexpected}")
    return values.astype(np.int8).to_numpy()


def read_selected_features(path: Path, expected_count: int) -> list[str]:
    selected = pd.read_csv(path, low_memory=False)
    if "feature" not in selected.columns:
        raise ValueError("The selected-descriptor file must contain a 'feature' column.")

    features = (
        selected["feature"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )
    if len(features) != expected_count:
        raise ValueError(f"Expected {expected_count} descriptors; found {len(features)}.")
    return features


def make_model(name: str, y_fit: np.ndarray, fold: int):
    """Create a fresh fold-specific model and preprocessing pipeline."""
    model_seed = MODEL_SEED + fold

    if name == "Logistic regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**LOGISTIC_PARAMS, random_state=model_seed)),
        ])

    if name == "Random forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**RANDOM_FOREST_PARAMS, random_state=model_seed)),
        ])

    if name == "Extra trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(**EXTRA_TREES_PARAMS, random_state=model_seed)),
        ])

    if name == "SVM-RBF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(**SVM_RBF_PARAMS, random_state=model_seed)),
        ])

    if name == "XGBoost":
        n_non_nades = int(np.sum(y_fit == 0))
        n_nades = int(np.sum(y_fit == 1))
        if n_non_nades == 0 or n_nades == 0:
            raise ValueError("Both classes must occur in every XGBoost fit fold.")

        xgb_model = xgb.XGBClassifier(
            **XGB_PARAMS,
            random_state=model_seed,
            scale_pos_weight=n_non_nades / n_nades,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb_model),
        ])

    raise ValueError(f"Unknown model name: {name}")


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Quantile-bin expected calibration error."""
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=float)
    edges = np.unique(np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return np.nan

    edges[0], edges[-1] = 0.0, 1.0
    bin_ids = np.digitize(probability, edges[1:-1], right=True)
    ece = 0.0

    for bin_index in range(len(edges) - 1):
        mask = bin_ids == bin_index
        if np.any(mask):
            ece += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(ece)


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "precision_nades": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_nades": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_nades": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity_non_nades": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_operating_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, dict]:
    """Maximum OOF balanced accuracy; NADES recall resolves exact ties."""
    selected_threshold = None
    selected_metrics = None
    selected_key = None

    for threshold in THRESHOLD_GRID:
        metrics = threshold_metrics(y_true, probability, float(threshold))
        key = (metrics["balanced_accuracy"], metrics["recall_nades"])
        if selected_key is None or key > selected_key:
            selected_threshold = float(threshold)
            selected_metrics = metrics
            selected_key = key

    return selected_threshold, selected_metrics


def complete_metrics(
    model_name: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict:
    record = {
        "model": model_name,
        "n_samples": int(len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "ap_nades": float(average_precision_score(y_true, probability)),
        "ap_non_nades": float(average_precision_score(1 - y_true, 1.0 - probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "ece_quantile_10": expected_calibration_error(y_true, probability),
    }
    record.update(threshold_metrics(y_true, probability, threshold))
    return record


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=12)


# ============================================================
# 3. LOAD THE FIXED TRAINING PARTITION ONLY
# ============================================================

for required_path in [DATA_PATH, SPLIT_PATH, SELECTED_FEATURES_PATH]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found:\n{required_path}")

data_all = pd.read_csv(
    DATA_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)
split_ref = pd.read_csv(SPLIT_PATH, low_memory=False)

if not {CODE_COLUMN, TARGET_COLUMN}.issubset(data_all.columns):
    raise ValueError("master_train_features.csv must contain Code and activity.")
if not {CODE_COLUMN, SPLIT_COLUMN}.issubset(split_ref.columns):
    raise ValueError("The fixed split file must contain Code and split.")

selected_features = read_selected_features(
    SELECTED_FEATURES_PATH,
    expected_count=SELECTED_SUBSET_SIZE,
)
missing_features = [feature for feature in selected_features if feature not in data_all.columns]
if missing_features:
    raise ValueError(f"Selected descriptors absent from data: {missing_features[:10]}")

data_all[CODE_COLUMN] = data_all[CODE_COLUMN].astype("string").str.strip()
split_ref[CODE_COLUMN] = split_ref[CODE_COLUMN].astype("string").str.strip()

if data_all[CODE_COLUMN].duplicated().any() or data_all[CODE_COLUMN].isna().any():
    raise ValueError("Master dataset must have unique, non-missing Code values.")
if split_ref[CODE_COLUMN].duplicated().any():
    raise ValueError("Split file contains duplicated Code values.")

split_map = split_ref.set_index(CODE_COLUMN)[SPLIT_COLUMN]
split_label = data_all[CODE_COLUMN].map(split_map)
if split_label.isna().any():
    examples = data_all.loc[split_label.isna(), CODE_COLUMN].tolist()[:10]
    raise ValueError(f"Codes missing from split file: {examples}")

train_mask = (split_label == TRAIN_SPLIT_LABEL).to_numpy()
if train_mask.sum() == 0:
    raise ValueError(
        f"No split == '{TRAIN_SPLIT_LABEL}' rows. "
        f"Found labels: {sorted(split_label.astype(str).unique().tolist())}"
    )

# Internal-evaluation samples are discarded before any OOF fitting begins.
train_data = data_all.loc[train_mask].copy().reset_index(drop=True)
del data_all

y_train = validate_binary_target(train_data[TARGET_COLUMN])
codes_train = train_data[CODE_COLUMN].astype(str).reset_index(drop=True)
X_train = (
    train_data[selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .reset_index(drop=True)
)
del train_data

if set(np.unique(y_train)) != {0, 1}:
    raise ValueError("The training partition must contain both classes.")

print("=" * 94)
print("MODEL-FAMILY BENCHMARK — TRAIN-ONLY OOF")
print("=" * 94)
print("Descriptor subset        :", len(selected_features))
print("Split source             :", SPLIT_PATH.name)
print("Training rows used       :", len(y_train))
print("Class balance            :", dict(pd.Series(y_train).value_counts().sort_index()))
print("Internal evaluation used : NO")
print("Blind data used          : NO")


# ============================================================
# 4. IDENTICAL FIVE-FOLD OOF PREDICTIONS FOR EVERY MODEL
# ============================================================

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_assignments = np.full(len(y_train), -1, dtype=int)
oof_probability = {
    model_name: np.full(len(y_train), np.nan, dtype=float)
    for model_name in MODEL_ORDER
}

for fold, (fit_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_fit = X_train.iloc[fit_idx]
    X_val = X_train.iloc[val_idx]
    y_fit = y_train[fit_idx]

    for model_name in MODEL_ORDER:
        model = make_model(model_name, y_fit, fold)
        model.fit(X_fit, y_fit)
        oof_probability[model_name][val_idx] = model.predict_proba(X_val)[:, 1]

    fold_assignments[val_idx] = fold
    print(f"Fold {fold}/{N_SPLITS} completed | fit n={len(fit_idx)} | validation n={len(val_idx)}")

if np.any(fold_assignments < 0):
    raise RuntimeError("Some samples did not receive an OOF fold assignment.")
for model_name, probability in oof_probability.items():
    if np.isnan(probability).any():
        raise RuntimeError(f"OOF probabilities were not filled for {model_name}.")


# ============================================================
# 5. OOF METRICS AND MODEL-SPECIFIC OPERATING THRESHOLDS
# ============================================================

metrics_rows = []
prediction_tables = []
threshold_map = {}

for model_name in MODEL_ORDER:
    probability = oof_probability[model_name]
    threshold, _ = select_operating_threshold(y_train, probability)
    threshold_map[model_name] = threshold
    metrics_rows.append(complete_metrics(model_name, y_train, probability, threshold))

    predicted = (probability >= threshold).astype(np.int8)
    prediction_tables.append(pd.DataFrame({
        "Code": codes_train,
        "fold": fold_assignments,
        "model": model_name,
        "true_activity": y_train,
        "true_class": np.where(y_train == 1, "NADES", "non-NADES"),
        "probability_NADES": probability,
        "probability_non_NADES": 1.0 - probability,
        "selected_oof_threshold": threshold,
        "predicted_activity": predicted,
        "predicted_class": np.where(predicted == 1, "NADES", "non-NADES"),
        "correct_prediction": predicted == y_train,
    }))

metrics_df = (
    pd.DataFrame(metrics_rows)
    .set_index("model")
    .loc[MODEL_ORDER]
    .reset_index()
)
metrics_df.to_csv(METRICS_OUT, index=False)
pd.concat(prediction_tables, ignore_index=True).to_csv(PREDICTIONS_OUT, index=False)

settings_df = pd.DataFrame([
    {
        "selected_subset_size": SELECTED_SUBSET_SIZE,
        "selected_features_file": data_path('DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv'),
        "n_train_only_samples": len(y_train),
        "n_non_nades": int(np.sum(y_train == 0)),
        "n_nades": int(np.sum(y_train == 1)),
        "n_splits": N_SPLITS,
        "split_seed": SEED,
        "threshold_selection": "maximum OOF balanced accuracy; NADES recall tie-breaker",
        "class_imbalance": "class_weight=balanced/balanced_subsample; XGBoost scale_pos_weight=n0/n1",
        "model_tuning": "none; fixed baseline parameters",
        "internal_evaluation_used": False,
        "blind_used": False,
        "interpretation": "train-only post-selection benchmark on the XGBoost-guided descriptor subset",
    }
])
settings_df.to_csv(SETTINGS_OUT, index=False)


# ============================================================
# 6. ROC AND PRECISION-RECALL CURVES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for model_name in MODEL_ORDER:
    probability = oof_probability[model_name]
    metrics = metrics_df.loc[metrics_df["model"] == model_name].iloc[0]
    color = MODEL_COLORS[model_name]

    fpr, tpr, _ = roc_curve(y_train, probability)
    axes[0].plot(
        fpr, tpr,
        color=color,
        linewidth=2.0 if model_name != "XGBoost" else 2.6,
        label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})",
    )

    precision, recall, _ = precision_recall_curve(y_train, probability)
    axes[1].plot(
        recall, precision,
        color=color,
        linewidth=2.0 if model_name != "XGBoost" else 2.6,
        label=f"{model_name} (AP = {metrics['ap_nades']:.3f})",
    )

axes[0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.2)
axes[1].axhline(y_train.mean(), linestyle="--", color="#999999", linewidth=1.2)

axes[0].set_xlabel("False Positive Rate", fontsize=14)
axes[0].set_ylabel("True Positive Rate", fontsize=14)
axes[1].set_xlabel("Recall", fontsize=14)
axes[1].set_ylabel("Precision", fontsize=14)

for ax in axes:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    style_axis(ax)

axes[0].legend(frameon=False, fontsize=9, loc="lower right")
axes[1].legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(ROC_PR_PNG, dpi=600, bbox_inches="tight")
fig.savefig(ROC_PR_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 7. RELIABILITY DIAGRAM
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 6.0))
ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.2, label="Perfect calibration")

for model_name in MODEL_ORDER:
    probability = oof_probability[model_name]
    metrics = metrics_df.loc[metrics_df["model"] == model_name].iloc[0]
    fraction_positive, mean_predicted = calibration_curve(
        y_train,
        probability,
        n_bins=10,
        strategy="quantile",
    )
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        linewidth=2.0 if model_name != "XGBoost" else 2.6,
        color=MODEL_COLORS[model_name],
        label=f"{model_name} (Brier = {metrics['brier_score']:.3f})",
    )

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.02)
ax.set_xlabel("Mean predicted probability", fontsize=14)
ax.set_ylabel("Observed NADES fraction", fontsize=14)
ax.legend(frameon=False, fontsize=9, loc="upper left")
style_axis(ax)
fig.tight_layout()
fig.savefig(RELIABILITY_PNG, dpi=600, bbox_inches="tight")
fig.savefig(RELIABILITY_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 94)
print("TRAIN-ONLY OOF MODEL-FAMILY BENCHMARK")
print("=" * 94)
print(
    metrics_df[[
        "model", "roc_auc", "ap_nades", "ap_non_nades", "brier_score",
        "ece_quantile_10", "threshold", "balanced_accuracy", "mcc",
        "specificity_non_nades", "recall_nades",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
)

print("\nSaved:")
for path in [
    METRICS_OUT,
    PREDICTIONS_OUT,
    SETTINGS_OUT,
    ROC_PR_PNG,
    ROC_PR_PDF,
    RELIABILITY_PNG,
    RELIABILITY_PDF,
]:
    print(" -", path)

print("\nNOTE: Internal-evaluation and blind data were not used.")
print("NOTE: This is a same-feature, post-selection development benchmark.")


