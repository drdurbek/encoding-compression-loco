
"""
NADES CROSS-TASK FAMILY READOUT — REPEATED TRAIN-ONLY SPLITS
=============================================================

Purpose
-------
This is a robustness experiment for the central representation claim:
which descriptor family retains the most endpoint-relevant information?

It does NOT replace either final model:
    * Final60 XGBoost classification remains the optimized screening model.
    * Final60 Random-Forest Tg model remains the optimized Tg model.

This script instead uses one controlled model family (Random Forest) to
compare descriptor families under repeated leakage-free validation.

Critical design choices
-----------------------
* Classification and Tg use DIFFERENT datasets and therefore DIFFERENT
  cross-fitting splitters:
      classification : RepeatedStratifiedKFold (5 folds x 3 repeats)
      Tg             : LeaveOneGroupOut by Core_NADES_ID (33 parent folds)
* The same fold is used for every family, so family differences are paired.
* Technical cleaning, supervised ranking, correlation pruning, imputation,
  and model fitting are all performed on that repetition's training rows only.
* Every family receives the same descriptor budget (K = 8).
* Optional A/B position-swap classification analysis uses exactly the same
  splits and readout procedure if the swapped master table is available.

Outputs
-------
CrossTask_Family_Multisplit_RF/
    per_fold_readouts.csv
    per_fold_predictions.csv
    per_fold_selected_features.csv
    per_fold_cleaning_log.csv
    paired_family_deltas.csv
    paired_delta_summary.csv
    classification_AB_swap_summary.csv       (if swapped input exists)
    family_readout_crossfitted.png/.pdf

Classification metrics are pooled over each complete five-fold repeat. Tg
metrics are pooled over all 33 LOGO predictions; parent-level RMSE/MAE values
are used for family contrasts. Quantiles across the three classification
repeats are descriptive sensitivity ranges, not confidence intervals. These
are cross-fitted robustness readouts, not replacements for the final held-out
or LOCO tests.
"""

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


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold


warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. PATHS — BOTH INPUT FILES ARE master_train_features.csv
# =============================================================================

BASE_CLASSIFICATION = data_path('')

BASE_TG = data_path('TG')

# Classification master table: NADES/non-NADES labels 0/1.
CLASSIFICATION_PATH = data_path('master_train_features.csv')

# Tg master table: continuous Tg values in activity, with Core_NADES_ID.
TG_PATH = data_path('TG/master_train_features.csv')

# Created by AB_Position_Swap_Control.py.  If it is absent, the primary
# original-encoding experiment still runs and the A/B control is skipped.
CLASSIFICATION_SWAP_PATH = (
    data_path('master_train_features_AB_position_swapped.csv')
)

OUTPUT_DIR = data_path('CrossTask_Family_Multisplit_RF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FIXED EXPERIMENT SETTINGS
# =============================================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"
TG_GROUP_COLUMN = "Core_NADES_ID"

CLASS_N_SPLITS = 5
CLASS_N_REPEATS = 3
DESCRIPTOR_BUDGET = 8

# This is a deterministic train-only ranking experiment, not a new GA sweep.
PREFILTER_TOP_N = 250
CORRELATION_THRESHOLD = 0.95
MAX_TRAIN_MISSING_FRACTION = 0.95

# The same Random-Forest family is used in both task modes. The classifier
# uses class_weight because non-NADES is the minority class.
RF_COMMON = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}

RF_CLASSIFIER_EXTRA = {
    "class_weight": "balanced",
}

RUN_AB_POSITION_SWAP_IF_AVAILABLE = True

plt.rcParams["font.family"] = "Arial"


# =============================================================================
# 3. FAMILY DEFINITIONS
# =============================================================================

# These prefix rules are shared between the classification and Tg encodings.
# A_ and B_ are anchored so AB_ interaction terms never enter Component_A/B.
FAMILY_ORDER = [
    "Component_A",
    "Component_B",
    "Component_A_plus_B",
    "Weighted_mixture",
    "Arithmetic_mean",
    "AB_interactions",
]

SHORT_LABELS = {
    "Component_A": "Component A",
    "Component_B": "Component B",
    "Component_A_plus_B": "A + B components",
    "Weighted_mixture": "Weighted mixture",
    "Arithmetic_mean": "Arithmetic mean",
    "AB_interactions": "AB interactions",
}


def family_of_column(column: str) -> str | None:
    if column.startswith("AB_"):
        return "AB_interactions"
    if column.startswith("mix_"):
        return "Weighted_mixture"
    if column.startswith("avg_"):
        return "Arithmetic_mean"
    if column.startswith("A_"):
        return "Component_A"
    if column.startswith("B_"):
        return "Component_B"
    return None


def build_family_map(columns: pd.Index) -> dict[str, list[str]]:
    family_map = {family: [] for family in FAMILY_ORDER}

    for column in columns:
        family = family_of_column(column)
        if family in {"Component_A", "Component_B", "Weighted_mixture", "Arithmetic_mean", "AB_interactions"}:
            family_map[family].append(column)

    family_map["Component_A_plus_B"] = (
        family_map["Component_A"] + family_map["Component_B"]
    )

    return family_map


# =============================================================================
# 4. DATA AND FEATURE HELPERS
# =============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


def load_table(path: Path, task: str) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)

    required = {CODE_COLUMN, TARGET_COLUMN}
    if task == "tg":
        required.add(TG_GROUP_COLUMN)

    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    if table[CODE_COLUMN].duplicated().any():
        examples = table.loc[table[CODE_COLUMN].duplicated(), CODE_COLUMN].head(10).tolist()
        raise ValueError(f"{path.name} has duplicated Code values: {examples}")

    table[TARGET_COLUMN] = pd.to_numeric(table[TARGET_COLUMN], errors="coerce")
    if table[TARGET_COLUMN].isna().any():
        raise ValueError(f"{path.name}: activity contains missing/non-numeric values.")

    if task == "classification":
        labels = set(table[TARGET_COLUMN].astype(int).unique())
        if labels != {0, 1}:
            raise ValueError(
                f"{path.name}: classification activity must be exactly {{0, 1}}, got {labels}."
            )
        table[TARGET_COLUMN] = table[TARGET_COLUMN].astype(int)
    else:
        table[TARGET_COLUMN] = table[TARGET_COLUMN].astype(float)
        if table[TG_GROUP_COLUMN].isna().any():
            raise ValueError(f"{path.name}: {TG_GROUP_COLUMN} contains missing values.")

    return table


def numeric_train_clean(
    train_frame: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Technical cleaning performed only on the outer-training rows."""
    X = train_frame.loc[:, candidate_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    log = {
        "input_columns": int(X.shape[1]),
        "removed_over_missing": 0,
        "removed_constant": 0,
        "removed_duplicate": 0,
        "retained_columns": 0,
    }

    keep_missing = X.isna().mean(axis=0) <= MAX_TRAIN_MISSING_FRACTION
    log["removed_over_missing"] = int((~keep_missing).sum())
    X = X.loc[:, keep_missing]

    keep_nonconstant = X.nunique(axis=0, dropna=True) > 1
    log["removed_constant"] = int((~keep_nonconstant).sum())
    X = X.loc[:, keep_nonconstant]

    # Exact duplicates are detected from TRAIN rows only.
    duplicate_mask = X.T.duplicated(keep="first")
    log["removed_duplicate"] = int(duplicate_mask.sum())
    X = X.loc[:, ~duplicate_mask]

    log["retained_columns"] = int(X.shape[1])
    return X, log


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def select_train_only_features(
    train_frame: pd.DataFrame,
    y_train: np.ndarray,
    candidate_columns: list[str],
    task: str,
    seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
    """
    Train-only cleaning -> univariate supervised prefilter -> correlation
    pruning -> fixed descriptor budget. No GA and no information from test rows.
    """
    X_clean, cleaning_log = numeric_train_clean(train_frame, candidate_columns)

    if X_clean.shape[1] < DESCRIPTOR_BUDGET:
        raise ValueError(
            f"Only {X_clean.shape[1]} technically clean columns remain, fewer than "
            f"the requested budget ({DESCRIPTOR_BUDGET})."
        )

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index,
    )

    y_train = np.asarray(y_train)

    if task == "classification":
        mask0 = y_train == 0
        mask1 = y_train == 1
        if mask0.sum() == 0 or mask1.sum() == 0:
            raise ValueError("Both classification classes must occur in every training split.")

        mean0 = X_imp.loc[mask0].mean(axis=0)
        mean1 = X_imp.loc[mask1].mean(axis=0)
        var0 = X_imp.loc[mask0].var(axis=0, ddof=1)
        var1 = X_imp.loc[mask1].var(axis=0, ddof=1)
        pooled_sd = np.sqrt((var0 + var1) / 2.0).replace(0.0, np.nan)
        screening_stat = ((mean1 - mean0).abs() / pooled_sd).fillna(0.0)
        screening_name = "abs_cohen_d"
    else:
        y_series = pd.Series(y_train, index=X_imp.index)
        screening_stat = X_imp.corrwith(y_series, method="spearman").abs().fillna(0.0)
        screening_name = "abs_spearman_rho"

    n_screen = min(PREFILTER_TOP_N, X_imp.shape[1])
    screen_columns = screening_stat.nlargest(n_screen).index.tolist()
    X_screen = X_imp.loc[:, screen_columns]

    if task == "classification":
        mi = mutual_info_classif(
            X_screen.to_numpy(),
            y_train,
            random_state=seed,
        )

        auc_star = []
        for column in screen_columns:
            auc = roc_auc_score(y_train, X_screen[column].to_numpy())
            auc_star.append(max(auc, 1.0 - auc))

        score = (
            0.50 * percentile_rank(screening_stat.loc[screen_columns].to_numpy())
            + 0.30 * percentile_rank(mi)
            + 0.20 * percentile_rank(np.asarray(auc_star))
        )

        score_frame = pd.DataFrame(
            {
                "feature": screen_columns,
                screening_name: screening_stat.loc[screen_columns].to_numpy(),
                "mutual_information": mi,
                "auc_star": auc_star,
                "prefilter_score": score,
            }
        )
    else:
        mi = mutual_info_regression(
            X_screen.to_numpy(),
            y_train,
            random_state=seed,
        )
        pearson_r2 = X_screen.corrwith(pd.Series(y_train, index=X_screen.index), method="pearson").pow(2).fillna(0.0)

        score = (
            0.50 * percentile_rank(screening_stat.loc[screen_columns].to_numpy())
            + 0.30 * percentile_rank(mi)
            + 0.20 * percentile_rank(pearson_r2.to_numpy())
        )

        score_frame = pd.DataFrame(
            {
                "feature": screen_columns,
                screening_name: screening_stat.loc[screen_columns].to_numpy(),
                "mutual_information": mi,
                "univariate_linear_r2": pearson_r2.to_numpy(),
                "prefilter_score": score,
            }
        )

    score_frame = score_frame.sort_values(
        ["prefilter_score", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    # Greedy correlation pruning preserves the best-ranked representative.
    ordered_columns = score_frame["feature"].tolist()
    correlation = X_screen.loc[:, ordered_columns].corr().abs().fillna(0.0)
    selected: list[str] = []

    for column in ordered_columns:
        if not selected:
            selected.append(column)
            continue
        if (correlation.loc[column, selected] < CORRELATION_THRESHOLD).all():
            selected.append(column)
        if len(selected) == DESCRIPTOR_BUDGET:
            break

    if len(selected) < DESCRIPTOR_BUDGET:
        raise ValueError(
            f"Correlation pruning retained only {len(selected)} descriptors; "
            f"the requested family budget is {DESCRIPTOR_BUDGET}."
        )

    score_frame["selected"] = score_frame["feature"].isin(selected)
    score_frame["selection_rank"] = np.nan
    rank_map = {feature: rank for rank, feature in enumerate(selected, start=1)}
    score_frame.loc[score_frame["selected"], "selection_rank"] = (
        score_frame.loc[score_frame["selected"], "feature"].map(rank_map)
    )

    return selected, score_frame, cleaning_log


def model_metrics_and_predictions(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    selected_features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit only on train rows; return fold metrics and cross-fitted predictions."""
    X_train = train_frame.loc[:, selected_features].apply(pd.to_numeric, errors="coerce")
    X_test = test_frame.loc[:, selected_features].apply(pd.to_numeric, errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    if task == "classification":
        model = RandomForestClassifier(
            random_state=seed,
            **RF_COMMON,
            **RF_CLASSIFIER_EXTRA,
        )
        model.fit(X_train_imp, y_train)

        probability_nades = model.predict_proba(X_test_imp)[:, 1]
        probability_non_nades = 1.0 - probability_nades
        prediction = (probability_nades >= 0.50).astype(int)

        metrics = {
            "roc_auc_nades": float(roc_auc_score(y_test, probability_nades)),
            "ap_non_nades": float(
                average_precision_score((np.asarray(y_test) == 0).astype(int), probability_non_nades)
            ),
            "mcc_at_0_50": float(matthews_corrcoef(y_test, prediction)),
        }
        predictions = pd.DataFrame(
            {
                "y_true": np.asarray(y_test),
                "probability_nades": probability_nades,
                "probability_non_nades": probability_non_nades,
                "prediction_at_0_50": prediction,
            },
            index=test_frame.index,
        )
        return metrics, predictions

    model = RandomForestRegressor(
        random_state=seed,
        **RF_COMMON,
    )
    model.fit(X_train_imp, y_train)
    prediction = model.predict(X_test_imp)

    metrics = {
        "r2": float(r2_score(y_test, prediction)),
        "rmse_K": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "mae_K": float(mean_absolute_error(y_test, prediction)),
        "mean_error_K": float(np.mean(prediction - np.asarray(y_test))),
    }
    predictions = pd.DataFrame(
        {
            "y_true": np.asarray(y_test),
            "prediction_Tg_K": prediction,
        },
        index=test_frame.index,
    )
    return metrics, predictions


# =============================================================================
# 5. CROSS-FITTING SPLITS
# =============================================================================

def make_classification_splits(y: np.ndarray) -> list[dict[str, object]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=CLASS_N_SPLITS,
        n_repeats=CLASS_N_REPEATS,
        random_state=20260801,
    )
    dummy = np.zeros(len(y))
    splits: list[dict[str, object]] = []
    for split_number, (train_index, test_index) in enumerate(splitter.split(dummy, y), start=1):
        repeat = (split_number - 1) // CLASS_N_SPLITS + 1
        fold = (split_number - 1) % CLASS_N_SPLITS + 1
        splits.append({"repeat": repeat, "fold": fold, "train": train_index, "test": test_index, "test_group": None})
    return splits


def make_tg_splits(y: np.ndarray, groups: np.ndarray) -> list[dict[str, object]]:
    splitter = LeaveOneGroupOut()
    dummy = np.zeros(len(y))
    splits: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(dummy, y, groups=groups), start=1):
        held_out_groups = np.unique(groups[test_index])
        if len(held_out_groups) != 1:
            raise RuntimeError("Each LeaveOneGroupOut fold must contain exactly one parent NADES.")
        splits.append({"repeat": 1, "fold": fold, "train": train_index, "test": test_index, "test_group": str(held_out_groups[0])})
    return splits


# =============================================================================
# 6. CROSS-FITTED FAMILY READOUT
# =============================================================================

def run_crossfitted_readout(table: pd.DataFrame, task: str, variant: str, splits: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_map = build_family_map(table.columns)
    missing_families = [family for family in FAMILY_ORDER if len(family_map[family]) == 0]
    if missing_families:
        raise ValueError(f"{task}/{variant}: no raw columns found for {missing_families}.")

    target = table[TARGET_COLUMN].to_numpy()
    readout_rows, selection_rows, cleaning_rows, prediction_rows = [], [], [], []
    print("\n" + "=" * 100)
    print(f"{task.upper()} | {variant} | {len(splits)} TRAIN-ONLY CROSS-FITTED FOLDS")
    print("=" * 100)

    for split in splits:
        repeat, fold = int(split["repeat"]), int(split["fold"])
        train_index, test_index = split["train"], split["test"]
        train_frame, test_frame = table.iloc[train_index].copy(), table.iloc[test_index].copy()
        y_train, y_test = target[train_index], target[test_index]
        print(f"Repeat {repeat} | fold {fold:02d}/{len(splits) if task == 'tg' else CLASS_N_SPLITS} | train={len(train_index)} | test={len(test_index)}")

        for family_number, family in enumerate(FAMILY_ORDER):
            selected, score_frame, cleaning_log = select_train_only_features(
                train_frame, y_train, family_map[family], task, 10000 * repeat + 100 * fold + family_number
            )
            metrics, predictions = model_metrics_and_predictions(
                train_frame, test_frame, selected, y_train, y_test, task, 100000 * repeat + 100 * fold + family_number
            )
            readout_rows.append({
                "task": task, "variant": variant, "repeat": repeat, "fold": fold,
                "test_group": split["test_group"], "family": family,
                "family_label": SHORT_LABELS[family], "n_train": len(train_index), "n_test": len(test_index),
                "descriptor_budget": DESCRIPTOR_BUDGET, **metrics,
            })
            score_frame = score_frame.copy()
            for name, value in [("task", task), ("variant", variant), ("repeat", repeat), ("fold", fold), ("family", family)]:
                score_frame.insert(len([c for c in ["task", "variant", "repeat", "fold", "family"] if c in score_frame.columns]), name, value)
            selection_rows.append(score_frame)
            cleaning_rows.append({"task": task, "variant": variant, "repeat": repeat, "fold": fold, "test_group": split["test_group"], "family": family, **cleaning_log})
            predictions = predictions.copy()
            predictions.insert(0, "task", task)
            predictions.insert(1, "variant", variant)
            predictions.insert(2, "repeat", repeat)
            predictions.insert(3, "fold", fold)
            predictions.insert(4, "test_group", split["test_group"])
            predictions.insert(5, "family", family)
            predictions.insert(6, CODE_COLUMN, test_frame[CODE_COLUMN].to_numpy())
            prediction_rows.append(predictions.reset_index(drop=True))

    return pd.DataFrame(readout_rows), pd.concat(selection_rows, ignore_index=True), pd.DataFrame(cleaning_rows), pd.concat(prediction_rows, ignore_index=True)


# =============================================================================
# 7. POOLED METRICS, PARENT ERRORS, AND PAIRED CONTRASTS
# =============================================================================

def classification_pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = predictions.loc[predictions["task"] == "classification"]
    for (variant, repeat, family), group in frame.groupby(["variant", "repeat", "family"], sort=False):
        if group[CODE_COLUMN].duplicated().any() or len(group) != group[CODE_COLUMN].nunique():
            raise RuntimeError("Each classification repeat must contain one OOF prediction per sample and family.")
        y = group["y_true"].to_numpy(dtype=int)
        p1 = group["probability_nades"].to_numpy(dtype=float)
        pred = group["prediction_at_0_50"].to_numpy(dtype=int)
        rows.append({
            "task": "classification", "variant": variant, "repeat": repeat, "family": family,
            "family_label": SHORT_LABELS[family], "n_samples": len(group),
            "roc_auc_nades": roc_auc_score(y, p1),
            "ap_non_nades": average_precision_score((y == 0).astype(int), 1.0 - p1),
            "mcc_at_0_50": matthews_corrcoef(y, pred),
        })
    return pd.DataFrame(rows)


def tg_pooled_and_parent_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows, parent_rows = [], []
    frame = predictions.loc[predictions["task"] == "tg"]
    for (variant, family), group in frame.groupby(["variant", "family"], sort=False):
        if group[CODE_COLUMN].duplicated().any() or len(group) != group[CODE_COLUMN].nunique():
            raise RuntimeError("LOGO must create one Tg prediction per sample and family.")
        y, p = group["y_true"].to_numpy(float), group["prediction_Tg_K"].to_numpy(float)
        pooled_rows.append({
            "task": "tg", "variant": variant, "family": family, "family_label": SHORT_LABELS[family],
            "n_samples": len(group), "r2": r2_score(y, p),
            "rmse_K": np.sqrt(mean_squared_error(y, p)), "mae_K": mean_absolute_error(y, p),
            "mean_error_K": np.mean(p - y),
        })
        for parent, parent_group in group.groupby("test_group", sort=False):
            yp, pp = parent_group["y_true"].to_numpy(float), parent_group["prediction_Tg_K"].to_numpy(float)
            parent_rows.append({
                "task": "tg", "variant": variant, "family": family, "family_label": SHORT_LABELS[family],
                "parent_nades": parent, "n_samples": len(parent_group),
                "rmse_K": np.sqrt(mean_squared_error(yp, pp)), "mae_K": mean_absolute_error(yp, pp),
                "mean_error_K": np.mean(pp - yp), "r2_within_parent": r2_score(yp, pp),
            })
    return pd.DataFrame(pooled_rows), pd.DataFrame(parent_rows)


COMPARISONS = {
    "tg": [("Weighted_mixture", "Arithmetic_mean"), ("Weighted_mixture", "AB_interactions"), ("Weighted_mixture", "Component_A"), ("Weighted_mixture", "Component_B")],
    "classification": [("Component_B", "Weighted_mixture"), ("Component_B", "Arithmetic_mean"), ("Component_B", "AB_interactions"), ("Component_A_plus_B", "Weighted_mixture")],
}


def paired_contrasts(class_pooled: pd.DataFrame, tg_parent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    # Three complete classification OOF repeats; descriptive repeat sensitivity only.
    for variant in class_pooled["variant"].unique():
        data = class_pooled.loc[class_pooled["variant"] == variant]
        for first, second in COMPARISONS["classification"]:
            left, right = data.loc[data.family == first].set_index("repeat"), data.loc[data.family == second].set_index("repeat")
            for repeat in left.index.intersection(right.index):
                for metric in ["roc_auc_nades", "ap_non_nades", "mcc_at_0_50"]:
                    rows.append({"task": "classification", "variant": variant, "unit": f"repeat_{repeat}", "first_family": first, "second_family": second, "comparison": f"{SHORT_LABELS[first]} vs {SHORT_LABELS[second]}", "metric": metric, "delta": left.loc[repeat, metric] - right.loc[repeat, metric], "positive_favors": first})
    # Tg: paired parent-level absolute errors; positive delta means lower error for first family.
    for variant in tg_parent["variant"].unique():
        data = tg_parent.loc[tg_parent["variant"] == variant]
        for first, second in COMPARISONS["tg"]:
            left, right = data.loc[data.family == first].set_index("parent_nades"), data.loc[data.family == second].set_index("parent_nades")
            for parent in left.index.intersection(right.index):
                for metric in ["rmse_K", "mae_K"]:
                    rows.append({"task": "tg", "variant": variant, "unit": parent, "first_family": first, "second_family": second, "comparison": f"{SHORT_LABELS[first]} vs {SHORT_LABELS[second]}", "metric": metric, "delta": right.loc[parent, metric] - left.loc[parent, metric], "positive_favors": first})
    deltas = pd.DataFrame(rows)
    summary = deltas.groupby(["task", "variant", "first_family", "second_family", "comparison", "metric", "positive_favors"], sort=False).agg(
        n_units=("delta", "size"), median_delta=("delta", "median"), mean_delta=("delta", "mean"),
        p2_5=("delta", lambda x: np.percentile(x, 2.5)), p97_5=("delta", lambda x: np.percentile(x, 97.5)),
        n_positive=("delta", lambda x: int((x > 0).sum())), n_negative=("delta", lambda x: int((x < 0).sum())),
    ).reset_index()
    return deltas, summary


# =============================================================================
# 8. FIGURES
# =============================================================================

def make_figures(class_pooled: pd.DataFrame, tg_parent: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    class_original = class_pooled.loc[class_pooled.variant == "original"]
    class_values = [class_original.loc[class_original.family == f, "roc_auc_nades"].to_numpy() for f in FAMILY_ORDER]
    tg_values = [tg_parent.loc[tg_parent.family == f, "rmse_K"].to_numpy() for f in FAMILY_ORDER]
    for ax, values, ylabel, note in [(axes[0], class_values, "Classification ROC–AUC", "3 complete repeated OOF passes"), (axes[1], tg_values, "Parent-level Tg RMSE (K)", "33 leave-one-parent-out folds")]:
        ax.boxplot(values, labels=[SHORT_LABELS[f] for f in FAMILY_ORDER], showfliers=False)
        for i, vals in enumerate(values, 1):
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=28, alpha=0.8, color="#298c8c")
        ax.set_ylabel(ylabel, fontsize=15); ax.tick_params(axis="x", rotation=35, labelsize=11); ax.tick_params(axis="y", labelsize=12)
        ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top", fontsize=11); ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].text(-0.16, 1.03, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=18)
    axes[1].text(-0.16, 1.03, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=18)
    fig.tight_layout(); fig.savefig(data_path('CrossTask_Family_Multisplit_RF/family_readout_crossfitted.png'), dpi=600, bbox_inches="tight"); fig.savefig(data_path('CrossTask_Family_Multisplit_RF/family_readout_crossfitted.pdf'), bbox_inches="tight"); plt.close(fig)


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:
    require_file(CLASSIFICATION_PATH, "Classification master file"); require_file(TG_PATH, "Tg master file")
    classification, tg = load_table(CLASSIFICATION_PATH, "classification"), load_table(TG_PATH, "tg")
    class_splits = make_classification_splits(classification[TARGET_COLUMN].to_numpy())
    tg_splits = make_tg_splits(tg[TARGET_COLUMN].to_numpy(), tg[TG_GROUP_COLUMN].astype(str).to_numpy())
    print("=" * 100); print("NADES CROSS-TASK FAMILY CROSS-FITTED RF READOUT"); print("=" * 100)
    print(f"Classification: {CLASS_N_SPLITS}-fold x {CLASS_N_REPEATS} repeats | Tg: LOGO across {len(tg_splits)} parent NADES | Budget: {DESCRIPTOR_BUDGET}")

    all_readouts, all_selections, all_cleaning, all_predictions = [], [], [], []
    for table, task, variant, splits in [(classification, "classification", "original", class_splits), (tg, "tg", "original", tg_splits)]:
        out = run_crossfitted_readout(table, task, variant, splits); all_readouts.append(out[0]); all_selections.append(out[1]); all_cleaning.append(out[2]); all_predictions.append(out[3])
    if RUN_AB_POSITION_SWAP_IF_AVAILABLE and CLASSIFICATION_SWAP_PATH.exists():
        swapped = load_table(CLASSIFICATION_SWAP_PATH, "classification")
        if not swapped[CODE_COLUMN].astype(str).equals(classification[CODE_COLUMN].astype(str)) or not swapped[TARGET_COLUMN].equals(classification[TARGET_COLUMN]):
            raise ValueError("A/B-swapped classification table must preserve original Code order and activity labels.")
        out = run_crossfitted_readout(swapped, "classification", "AB_position_swapped", class_splits); all_readouts.append(out[0]); all_selections.append(out[1]); all_cleaning.append(out[2]); all_predictions.append(out[3])
    else:
        print("[INFO] A/B position-swap file absent: swap control skipped.")

    readouts, selections, cleaning, predictions = (pd.concat(all_readouts, ignore_index=True), pd.concat(all_selections, ignore_index=True), pd.concat(all_cleaning, ignore_index=True), pd.concat(all_predictions, ignore_index=True))
    class_pooled = classification_pooled_metrics(predictions); tg_pooled, tg_parent = tg_pooled_and_parent_metrics(predictions); deltas, delta_summary = paired_contrasts(class_pooled, tg_parent)
    readouts.to_csv(data_path('CrossTask_Family_Multisplit_RF/per_fold_readouts.csv'), index=False); predictions.to_csv(data_path('CrossTask_Family_Multisplit_RF/per_fold_predictions.csv'), index=False); selections.to_csv(data_path('CrossTask_Family_Multisplit_RF/per_fold_selected_features.csv'), index=False); cleaning.to_csv(data_path('CrossTask_Family_Multisplit_RF/per_fold_cleaning_log.csv'), index=False)
    class_pooled.to_csv(data_path('CrossTask_Family_Multisplit_RF/classification_pooled_repeat_metrics.csv'), index=False); tg_pooled.to_csv(data_path('CrossTask_Family_Multisplit_RF/tg_logo_pooled_metrics.csv'), index=False); tg_parent.to_csv(data_path('CrossTask_Family_Multisplit_RF/tg_logo_parent_metrics.csv'), index=False); deltas.to_csv(data_path('CrossTask_Family_Multisplit_RF/paired_family_deltas.csv'), index=False); delta_summary.to_csv(data_path('CrossTask_Family_Multisplit_RF/paired_delta_summary.csv'), index=False)
    make_figures(class_pooled, tg_parent)
    print("\nCOMPLETED. Classification pooled metrics:"); print(class_pooled.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTg LOGO pooled metrics:"); print(tg_pooled.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPaired contrasts (positive favors first family):"); print(delta_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


