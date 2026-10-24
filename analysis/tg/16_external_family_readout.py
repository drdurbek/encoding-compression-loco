
"""
TG FAMILY READOUT — FROZEN BLIND2 VALIDATION
=============================================

Adapted directly from TG_Family_Blind_Concentration_Validation_RF.py.

It preserves the same controlled family-readout procedure:
    * K = 8 descriptors per family;
    * training-only technical filtering, supervised prefilter, and correlation pruning;
    * RandomForestRegressor (300 trees, max_depth=6);
    * one model per descriptor family trained on all 263 labelled Tg samples.

Differences for blind2:
    * input is TG/blind2/master_blind2_features.csv;
    * true Tg values are read only AFTER the six blind prediction vectors
      have been constructed;
    * output contains frozen predictions and blind2 RMSE/MAE/R2 for all six
      family models.

This is not the Final60 deployment prediction. It is the controlled
family-readout blind validation needed to compare representations.
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. PATHS
# =============================================================================

BASE = data_path('TG')

TRAIN_PATH = data_path('TG/master_train_features.csv')
BLIND_FEATURE_PATH = data_path('TG/blind2/master_blind2_features.csv')
BLIND_LABEL_PATH = data_path('TG/blind2/blind2_with_activity.csv')
OUTPUT_DIR = data_path('TG/blind2/TG_Family_Blind2_Validation_RF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_OUT = data_path('TG/blind2/TG_Family_Blind2_Validation_RF/blind2_family_metrics.csv')
PREDICTIONS_LONG_OUT = data_path('TG/blind2/TG_Family_Blind2_Validation_RF/blind2_family_predictions_long.csv')
PREDICTIONS_WIDE_OUT = data_path('TG/blind2/TG_Family_Blind2_Validation_RF/blind2_family_predictions_wide.csv')
SELECTED_FEATURES_OUT = data_path('TG/blind2/TG_Family_Blind2_Validation_RF/blind2_family_selected_features.csv')
CLEANING_OUT = data_path('TG/blind2/TG_Family_Blind2_Validation_RF/blind2_family_cleaning_log.csv')


# =============================================================================
# 2. FROZEN FAMILY-READOUT SETTINGS — UNCHANGED
# =============================================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"
DESCRIPTOR_BUDGET = 8
PREFILTER_TOP_N = 250
CORRELATION_THRESHOLD = 0.95
MAX_TRAIN_MISSING_FRACTION = 0.95
SEED = 20260801

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}

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


# =============================================================================
# 3. UNCHANGED FAMILY AND TRAIN-ONLY SELECTION FUNCTIONS
# =============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


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
    result = {family: [] for family in FAMILY_ORDER}
    for column in columns:
        family = family_of_column(column)
        if family in {
            "Component_A", "Component_B", "Weighted_mixture",
            "Arithmetic_mean", "AB_interactions",
        }:
            result[family].append(column)
    result["Component_A_plus_B"] = result["Component_A"] + result["Component_B"]
    return result


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def clean_train_features(
    train: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    X = train.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    log = {"input_columns": int(X.shape[1])}

    keep = X.isna().mean(axis=0) <= MAX_TRAIN_MISSING_FRACTION
    log["removed_over_missing"] = int((~keep).sum())
    X = X.loc[:, keep]
    keep = X.nunique(axis=0, dropna=True) > 1
    log["removed_constant"] = int((~keep).sum())
    X = X.loc[:, keep]
    duplicate = X.T.duplicated(keep="first")
    log["removed_duplicate"] = int(duplicate.sum())
    X = X.loc[:, ~duplicate]
    log["retained_columns"] = int(X.shape[1])
    return X, log


def select_features_train_only(
    train: pd.DataFrame,
    y: np.ndarray,
    candidate_columns: list[str],
    seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
    """Same deterministic Tg family selection as in the blind1 readout script."""
    X_clean, log = clean_train_features(train, candidate_columns)
    if X_clean.shape[1] < DESCRIPTOR_BUDGET:
        raise ValueError(f"Only {X_clean.shape[1]} clean features remain; need K={DESCRIPTOR_BUDGET}.")

    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index,
    )
    y_series = pd.Series(y, index=X.index)
    abs_spearman = X.corrwith(y_series, method="spearman").abs().fillna(0.0)
    screen_columns = abs_spearman.nlargest(min(PREFILTER_TOP_N, X.shape[1])).index.tolist()
    X_screen = X.loc[:, screen_columns]
    mutual_information = mutual_info_regression(X_screen.to_numpy(), y, random_state=seed)
    pearson_r2 = X_screen.corrwith(y_series, method="pearson").pow(2).fillna(0.0)
    score = (
        0.50 * percentile_rank(abs_spearman.loc[screen_columns].to_numpy())
        + 0.30 * percentile_rank(mutual_information)
        + 0.20 * percentile_rank(pearson_r2.to_numpy())
    )
    scores = pd.DataFrame({
        "feature": screen_columns,
        "abs_spearman_rho": abs_spearman.loc[screen_columns].to_numpy(),
        "mutual_information": mutual_information,
        "univariate_linear_r2": pearson_r2.to_numpy(),
        "prefilter_score": score,
    }).sort_values(["prefilter_score", "feature"], ascending=[False, True]).reset_index(drop=True)

    ordered = scores["feature"].tolist()
    correlation = X_screen.loc[:, ordered].corr().abs().fillna(0.0)
    selected: list[str] = []
    for column in ordered:
        if not selected or (correlation.loc[column, selected] < CORRELATION_THRESHOLD).all():
            selected.append(column)
        if len(selected) == DESCRIPTOR_BUDGET:
            break
    if len(selected) != DESCRIPTOR_BUDGET:
        raise ValueError(f"Correlation pruning retained {len(selected)}, not K={DESCRIPTOR_BUDGET}.")

    rank_map = {feature: rank for rank, feature in enumerate(selected, start=1)}
    scores["selected"] = scores["feature"].isin(selected)
    scores["selection_rank"] = scores["feature"].map(rank_map)
    return selected, scores, log


# =============================================================================
# 4. TRAIN ON ALL LABELLED Tg DATA; PREDICT BLIND2
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_K": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(y_pred - y_true)),
    }


def main() -> None:
    require_file(TRAIN_PATH, "Labelled Tg master file")
    require_file(BLIND_FEATURE_PATH, "blind2 feature matrix")
    require_file(BLIND_LABEL_PATH, "blind2 Tg label file")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    blind = pd.read_csv(BLIND_FEATURE_PATH, low_memory=False)
    if {CODE_COLUMN, TARGET_COLUMN} - set(train.columns):
        raise ValueError("Training Tg master must contain Code and activity.")
    if CODE_COLUMN not in blind.columns or blind[CODE_COLUMN].isna().any() or blind[CODE_COLUMN].duplicated().any():
        raise ValueError("blind2 feature matrix must contain unique, non-missing Code values.")
    if TARGET_COLUMN in blind.columns:
        raise ValueError("Blind2 feature matrix must not contain experimental activity/Tg values.")

    y_train = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training Tg values must be numeric and non-missing.")
    y_train = y_train.to_numpy(float)

    family_map = build_family_map(train.columns)
    model_columns = set().union(*family_map.values())
    missing = sorted(model_columns - set(blind.columns))
    if missing:
        raise ValueError(
            f"blind2 feature matrix lacks {len(missing)} expected descriptor columns, e.g. {missing[:15]}"
        )

    print("=" * 100)
    print("TG FAMILY READOUT — FROZEN BLIND2 VALIDATION")
    print("=" * 100)
    print(f"Labelled training rows : {len(train)}")
    print(f"Blind2 mixtures        : {len(blind)}")
    print(f"Descriptor budget      : {DESCRIPTOR_BUDGET} per family")
    print("Blind2 Tg values are not accessed until all predictions are generated.\n")

    long_predictions, selected_tables, cleaning_rows, wide_columns = [], [], [], {CODE_COLUMN: blind[CODE_COLUMN].to_numpy()}

    for family_number, family in enumerate(FAMILY_ORDER):
        seed = SEED + family_number
        selected, scores, cleaning = select_features_train_only(train, y_train, family_map[family], seed)

        X_train = train.loc[:, selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_blind = blind.loc[:, selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_blind = imputer.transform(X_blind)

        model = RandomForestRegressor(random_state=seed, **RF_PARAMS)
        model.fit(X_train, y_train)
        prediction = model.predict(X_blind)

        column_name = f"predicted_Tg_K__{family}"
        wide_columns[column_name] = prediction
        long_predictions.append(pd.DataFrame({
            CODE_COLUMN: blind[CODE_COLUMN].to_numpy(),
            "family": family,
            "family_label": SHORT_LABELS[family],
            "n_selected_descriptors": len(selected),
            "predicted_Tg_K": prediction,
        }))
        scores.insert(0, "family", family)
        scores.insert(1, "family_label", SHORT_LABELS[family])
        selected_tables.append(scores)
        cleaning_rows.append({
            "family": family,
            "family_label": SHORT_LABELS[family],
            "n_selected_descriptors": len(selected),
            **cleaning,
        })
        print(f"{SHORT_LABELS[family]:22s} | selected {len(selected)} | predictions complete")

    long_frame = pd.concat(long_predictions, ignore_index=True)
    wide_frame = pd.DataFrame(wide_columns)
    selected_frame = pd.concat(selected_tables, ignore_index=True)
    cleaning_frame = pd.DataFrame(cleaning_rows)

    # -------------------------------------------------------------------------
    # The experimental blind2 Tg values enter only here, after all six family
    # prediction vectors have been generated.
    # -------------------------------------------------------------------------
    blind_labels = pd.read_csv(BLIND_LABEL_PATH, low_memory=False)
    if not {CODE_COLUMN, TARGET_COLUMN}.issubset(blind_labels.columns):
        raise ValueError("blind2_with_activity.csv must contain Code and activity.")
    blind_labels = blind_labels[[CODE_COLUMN, TARGET_COLUMN]].copy()
    blind_labels[TARGET_COLUMN] = pd.to_numeric(blind_labels[TARGET_COLUMN], errors="coerce")
    if blind_labels[CODE_COLUMN].duplicated().any() or blind_labels[TARGET_COLUMN].isna().any():
        raise ValueError("blind2 labels must have unique Code values and numeric Tg values.")
    if set(blind[CODE_COLUMN].astype(str)) != set(blind_labels[CODE_COLUMN].astype(str)):
        raise ValueError("blind2 feature and label Code sets do not match exactly.")

    long_frame[CODE_COLUMN] = long_frame[CODE_COLUMN].astype(str)
    blind_labels[CODE_COLUMN] = blind_labels[CODE_COLUMN].astype(str)
    long_frame = long_frame.merge(blind_labels, on=CODE_COLUMN, how="left", validate="many_to_one")
    long_frame["error_K"] = long_frame["predicted_Tg_K"] - long_frame[TARGET_COLUMN]
    long_frame["absolute_error_K"] = long_frame["error_K"].abs()
    long_frame = long_frame.sort_values(["family", CODE_COLUMN]).reset_index(drop=True)

    metric_rows = []
    for family in FAMILY_ORDER:
        frame = long_frame.loc[long_frame["family"] == family]
        metric_rows.append({
            "family": family,
            "family_label": SHORT_LABELS[family],
            "n_blind": len(frame),
            **regression_metrics(
                frame[TARGET_COLUMN].to_numpy(float),
                frame["predicted_Tg_K"].to_numpy(float),
            ),
        })
    metrics = pd.DataFrame(metric_rows)

    metrics.to_csv(METRICS_OUT, index=False)
    long_frame.to_csv(PREDICTIONS_LONG_OUT, index=False)
    wide_frame.to_csv(PREDICTIONS_WIDE_OUT, index=False)
    selected_frame.to_csv(SELECTED_FEATURES_OUT, index=False)
    cleaning_frame.to_csv(CLEANING_OUT, index=False)

    print("\n" + "=" * 100)
    print("BLIND2 FAMILY RESULTS")
    print("=" * 100)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved:")
    for path in [METRICS_OUT, PREDICTIONS_LONG_OUT, PREDICTIONS_WIDE_OUT, SELECTED_FEATURES_OUT, CLEANING_OUT]:
        print(" -", path)
    print("\nThis blind2 set is now consumed for this family-level validation.")
    print("Do not use its results to alter descriptor budget, family definitions, or RF parameters.")


if __name__ == "__main__":
    main()

