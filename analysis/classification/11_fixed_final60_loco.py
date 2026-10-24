
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
# LEAVE-ONE-COMPONENT-OUT (LOCO) VALIDATION
# FINAL 60-DESCRIPTOR XGBOOST MODEL
#
# Purpose
# -------
# This is a complementary chemical-extrapolation validation.
# For each component, every mixture containing that component is
# held out; the model is fitted only on mixtures without it.
#
# The 60 descriptors and tau = 0.42 must already be frozen by the
# train-only DAGA/OOF workflow. They are NOT selected or tuned here.
# The blind data are never loaded.
#
# Outputs
# -------
#   Final60_LOCO_component_results.csv
#   Final60_LOCO_sample_predictions.csv
#   Final60_LOCO_AL_candidates.csv       (only if requested components exist)
#   Final60_LOCO_summary.csv
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
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

MASTER_PATH = data_path('master_train_features.csv')
ORIGINAL_PATH = data_path('original.csv')

# Fixed descriptor set selected by the clean train-only DAGA run.
SELECTED_FEATURES_PATH = (
    data_path('DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('Final60_LOCO')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = data_path('Final60_LOCO/Final60_LOCO_component_results.csv')
PREDICTIONS_OUT = data_path('Final60_LOCO/Final60_LOCO_sample_predictions.csv')
AL_CANDIDATES_OUT = data_path('Final60_LOCO/Final60_LOCO_AL_candidates.csv')
SUMMARY_OUT = data_path('Final60_LOCO/Final60_LOCO_summary.csv')


# ============================================================
# 2. FROZEN SETTINGS
# ============================================================

MODEL_SEED = 82
THRESHOLD = 0.42                 # Frozen from train-only DAGA OOF.
EXPECTED_DESCRIPTOR_COUNT = 60
MIN_TEST_SAMPLES = 5
TOP_AL_PER_COMPONENT = 10

# Optional: export samples nearest tau for these held-out components.
# Set to an empty set if no active-learning table is needed.
AL_COMPONENTS = {
    "Maltose",
    "Malicacid",
    "Malonicacid",
}

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

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

COMPONENT_COLUMNS = ["A_id", "B_id", "C_id", "D_id"]


# ============================================================
# 3. HELPERS
# ============================================================

def load_selected_descriptors(path: Path) -> list[str]:
    """Load a unique, fixed Final60 descriptor list from a CSV."""
    table = pd.read_csv(path)

    supported_names = [
        "feature",
        "descriptor",
        "selected_descriptor",
        "selected_descriptors",
        "Feature",
        "Descriptor",
    ]

    descriptor_column = next(
        (column for column in supported_names if column in table.columns),
        None,
    )

    if descriptor_column is None:
        if table.shape[1] != 1:
            raise ValueError(
                "Cannot identify the descriptor column in "
                f"{path.name}. Columns: {table.columns.tolist()}"
            )
        descriptor_column = table.columns[0]

    descriptors = (
        table[descriptor_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    descriptors = list(
        dict.fromkeys(
            descriptor for descriptor in descriptors if descriptor
        )
    )

    if len(descriptors) != EXPECTED_DESCRIPTOR_COUNT:
        raise ValueError(
            "The Final60 descriptor file must contain exactly "
            f"{EXPECTED_DESCRIPTOR_COUNT} unique descriptors; found "
            f"{len(descriptors)}."
        )

    return descriptors


def class_counts(values: np.ndarray) -> tuple[int, int]:
    """Return (non-NADES, NADES) counts for binary labels."""
    return int(np.sum(values == 0)), int(np.sum(values == 1))


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True-negative rate for non-NADES; NaN if no negatives exist."""
    tn, fp, _, _ = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else np.nan


def build_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    """Build the fixed Final60 XGBoost classifier for one LOCO fold."""
    return xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )


# ============================================================
# 4. LOAD LABELLED DATA AND THE FIXED DESCRIPTOR SET
# ============================================================

master = pd.read_csv(
    MASTER_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

original = pd.read_csv(
    ORIGINAL_PATH,
    low_memory=False,
)

required_master_columns = {"Code", "activity"}
required_original_columns = {"Code", *COMPONENT_COLUMNS}

missing_master = required_master_columns - set(master.columns)
missing_original = required_original_columns - set(original.columns)

if missing_master:
    raise ValueError(
        "master_train_features.csv is missing: "
        f"{sorted(missing_master)}"
    )

if missing_original:
    raise ValueError(
        "original.csv is missing: "
        f"{sorted(missing_original)}"
    )

if master["Code"].duplicated().any():
    raise ValueError("Duplicate Code values in master_train_features.csv.")

if original["Code"].duplicated().any():
    raise ValueError("Duplicate Code values in original.csv.")

df = master.merge(
    original[["Code", *COMPONENT_COLUMNS]],
    on="Code",
    how="inner",
    validate="one_to_one",
)

if len(df) != len(master):
    missing_codes = sorted(
        set(master["Code"].astype(str)) - set(df["Code"].astype(str))
    )
    raise ValueError(
        "Some labelled rows do not have component identities in original.csv. "
        f"Missing = {len(missing_codes)}; examples = {missing_codes[:10]}"
    )

y_numeric = pd.to_numeric(df["activity"], errors="coerce")
if y_numeric.isna().any():
    raise ValueError("activity contains missing or non-numeric values.")

y = y_numeric.astype(np.int8).reset_index(drop=True)
if set(y.unique()) - {0, 1}:
    raise ValueError("activity must be binary: 0 = non-NADES; 1 = NADES.")

selected_features = load_selected_descriptors(SELECTED_FEATURES_PATH)

missing_features = [
    feature for feature in selected_features if feature not in df.columns
]
if missing_features:
    raise ValueError(
        "Selected Final60 descriptors are absent from master data: "
        f"{missing_features}"
    )

X_raw = (
    df.loc[:, selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .reset_index(drop=True)
)

codes = df["Code"].astype(str).reset_index(drop=True)

# Strip component names, retaining missing entries as pandas <NA>.
component_frame = df.loc[:, COMPONENT_COLUMNS].copy()
for column in COMPONENT_COLUMNS:
    component_frame[column] = (
        component_frame[column].astype("string").str.strip()
    )

components = sorted(
    str(component)
    for component in pd.unique(
        pd.concat(
            [component_frame[column] for column in COMPONENT_COLUMNS],
            ignore_index=True,
        ).dropna()
    )
    if str(component)
)

print("=" * 94)
print("FINAL60 LEAVE-ONE-COMPONENT-OUT (LOCO) VALIDATION")
print("=" * 94)
print(f"Labelled samples      : {len(df)}")
print(f"Fixed descriptors     : {len(selected_features)}")
print(f"Unique components     : {len(components)}")
print(f"Frozen threshold      : {THRESHOLD:.2f}")
print("Selection rule        : descriptors and threshold are fixed before LOCO")


# ============================================================
# 5. LOCO VALIDATION
# ============================================================

component_results: list[dict] = []
sample_prediction_tables: list[pd.DataFrame] = []
al_candidate_tables: list[pd.DataFrame] = []

for component_index, component in enumerate(components, start=1):
    # Every mixture containing the component is the held-out LOCO fold.
    component_mask = np.zeros(len(df), dtype=bool)
    for column in COMPONENT_COLUMNS:
        component_mask |= component_frame[column].eq(component).fillna(False).to_numpy()

    test_indices = np.flatnonzero(component_mask)
    train_indices = np.flatnonzero(~component_mask)

    n_test = len(test_indices)
    n_train = len(train_indices)

    if n_test < MIN_TEST_SAMPLES:
        continue

    X_train_raw = X_raw.iloc[train_indices]
    X_test_raw = X_raw.iloc[test_indices]
    y_train = y.iloc[train_indices].to_numpy(dtype=np.int8)
    y_test = y.iloc[test_indices].to_numpy(dtype=np.int8)

    n_train_negative, n_train_positive = class_counts(y_train)
    n_test_negative, n_test_positive = class_counts(y_test)

    if n_train_negative == 0 or n_train_positive == 0:
        print(
            f"Skipped {component}: LOCO training data do not contain both classes."
        )
        continue

    # Fit all learned preprocessing on the current LOCO training fold only.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw).astype(np.float32)
    X_test = imputer.transform(X_test_raw).astype(np.float32)

    model = build_model(
        scale_pos_weight=n_train_negative / n_train_positive
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(np.int8)

    # ROC-AUC, AP, balanced accuracy and MCC are not meaningful for a
    # held-out component containing a single observed class.
    has_both_test_classes = (
        n_test_negative > 0 and n_test_positive > 0
    )

    if has_both_test_classes:
        roc_auc = roc_auc_score(y_test, probabilities)
        pr_auc_nades = average_precision_score(y_test, probabilities)
        pr_auc_non_nades = average_precision_score(
            1 - y_test,
            1.0 - probabilities,
        )
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        mcc = matthews_corrcoef(y_test, predictions)
    else:
        roc_auc = np.nan
        pr_auc_nades = np.nan
        pr_auc_non_nades = np.nan
        balanced_accuracy = np.nan
        mcc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "component": component,
        "n_train": n_train,
        "n_test": n_test,
        "n_train_non_NADES": n_train_negative,
        "n_train_NADES": n_train_positive,
        "n_test_non_NADES": n_test_negative,
        "n_test_NADES": n_test_positive,
        "threshold": THRESHOLD,
        "ROC_AUC": float(roc_auc),
        "PR_AUC_NADES": float(pr_auc_nades),
        "PR_AUC_non_NADES": float(pr_auc_non_nades),
        "PR_baseline_NADES": float(n_test_positive / n_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy),
        "MCC": float(mcc),
        "specificity_non_NADES": float(specificity(y_test, predictions)),
        "precision_NADES": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_NADES": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "F1_NADES": float(f1_score(y_test, predictions, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    component_results.append(result)

    component_predictions = pd.DataFrame(
        {
            "Code": codes.iloc[test_indices].to_numpy(),
            "held_out_component": component,
            "true_activity": y_test,
            "true_class": np.where(y_test == 1, "NADES", "non-NADES"),
            "probability_NADES": probabilities,
            "probability_non_NADES": 1.0 - probabilities,
            "threshold_used": THRESHOLD,
            "predicted_activity": predictions,
            "predicted_class": np.where(
                predictions == 1, "NADES", "non-NADES"
            ),
            "correct_prediction": predictions == y_test,
            "distance_to_threshold": np.abs(probabilities - THRESHOLD),
            "signed_margin": probabilities - THRESHOLD,
        }
    )
    sample_prediction_tables.append(component_predictions)

    if component in AL_COMPONENTS:
        candidates = (
            component_predictions
            .sort_values(["distance_to_threshold", "Code"])
            .head(TOP_AL_PER_COMPONENT)
            .copy()
        )
        candidates.insert(2, "AL_rank", np.arange(1, len(candidates) + 1))
        al_candidate_tables.append(candidates)

    print(
        f"{component_index:>3}/{len(components)} | {component:<30} | "
        f"n_test={n_test:>3} | ROC-AUC={roc_auc:.3f} | "
        f"PR-AUC={pr_auc_nades:.3f} | BAcc={balanced_accuracy:.3f} | "
        f"MCC={mcc:.3f}"
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

if not component_results:
    raise RuntimeError("No components met the LOCO inclusion criterion.")

results_df = (
    pd.DataFrame(component_results)
    .sort_values(
        ["ROC_AUC", "PR_AUC_NADES", "n_test"],
        ascending=[True, True, False],
        na_position="last",
    )
    .reset_index(drop=True)
)
results_df.to_csv(RESULTS_OUT, index=False)

predictions_df = pd.concat(sample_prediction_tables, ignore_index=True)
predictions_df.to_csv(PREDICTIONS_OUT, index=False)

if al_candidate_tables:
    al_candidates_df = pd.concat(al_candidate_tables, ignore_index=True)
    al_candidates_df.to_csv(AL_CANDIDATES_OUT, index=False)
else:
    al_candidates_df = pd.DataFrame()

summary_metrics = [
    "ROC_AUC",
    "PR_AUC_NADES",
    "PR_AUC_non_NADES",
    "accuracy",
    "balanced_accuracy",
    "MCC",
    "specificity_non_NADES",
    "precision_NADES",
    "recall_NADES",
    "F1_NADES",
]

summary_rows = []
for metric in summary_metrics:
    values = pd.to_numeric(results_df[metric], errors="coerce").dropna()
    summary_rows.append(
        {
            "metric": metric,
            "n_valid_components": int(len(values)),
            "mean": float(values.mean()) if len(values) else np.nan,
            "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            "median": float(values.median()) if len(values) else np.nan,
            "minimum": float(values.min()) if len(values) else np.nan,
            "maximum": float(values.max()) if len(values) else np.nan,
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False)


# ============================================================
# 7. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 94)
print("FINAL60 LOCO VALIDATION")
print("=" * 94)
print(f"Evaluated components             : {len(results_df)}")
print(
    "Components with both test classes: "
    f"{results_df['ROC_AUC'].notna().sum()}"
)
print(f"Mean ROC-AUC                     : {results_df['ROC_AUC'].mean(skipna=True):.4f}")
print(f"Mean AP for NADES                : {results_df['PR_AUC_NADES'].mean(skipna=True):.4f}")
print(f"Mean balanced accuracy            : {results_df['balanced_accuracy'].mean(skipna=True):.4f}")
print(f"Mean MCC                         : {results_df['MCC'].mean(skipna=True):.4f}")

print("\nTen weakest components by ROC-AUC:")
print(
    results_df[
        [
            "component",
            "n_test",
            "n_test_non_NADES",
            "n_test_NADES",
            "ROC_AUC",
            "PR_AUC_NADES",
            "balanced_accuracy",
            "MCC",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

if not al_candidates_df.empty:
    print(
        "\nActive-learning candidates exported for: "
        + ", ".join(sorted(AL_COMPONENTS))
    )

print("\nSaved:")
for path in [RESULTS_OUT, PREDICTIONS_OUT, SUMMARY_OUT]:
    print(" -", path)
if not al_candidates_df.empty:
    print(" -", AL_CANDIDATES_OUT)

print("\nBlind data were not loaded or used.")




