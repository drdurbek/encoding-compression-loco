

"""
TG PURE-NADES CONTROL: TRAIN-ONLY LOGO + POST-HOC BLIND2 DIAGNOSTIC
===================================================================

Purpose
-------
Test the chemical hypothesis that the blind2 failure is driven partly by a
water-content domain shift.  The model is rebuilt using ONLY labelled rows
whose water_frc is exactly zero (100% NADES), then evaluated by leave-one-
parent-NADES-out (LOGO) and used to predict the dry blind2 systems.

Critical interpretation rule
----------------------------
blind2 labels have already been inspected in the preceding aqueous-model
experiment.  Therefore, blind2 results produced here are a post-hoc
mechanistic diagnostic, NOT a new independent external-validation claim.
No descriptor budget, family definition, model setting, or threshold may be
changed after inspecting this script's blind2 output.

Protocol
--------
* water_frc == 0 only; no further descriptor rescaling is needed because the
  existing master feature matrix was multiplied by (1 - water_frc), which is
  one for anhydrous NADES;
* K = 8 descriptors per family;
* technical cleaning, supervised prefiltering, and correlation pruning occur
  inside every LOGO training fold;
* Random Forest and RidgeCV are readouts of the same selected descriptors;
* all parent-NADES groups are held out once in pooled LOGO predictions;
* blind2 labels are read only after all pure-model blind2 predictions exist;
* blind2 uncertainty uses parent-group bootstrap and compares pure-model
  predictions with the frozen aqueous-training predictions already on disk.

Required input files
--------------------
TG/
    master_train_features.csv
    original_water_frc.csv                 # Code, water_frc
    TG_core_nades_groups.csv                # Code, Core_NADES_ID
    blind2/
        master_blind2_features.csv
        blind2_with_activity.csv
        TG_Family_Blind2_Robust_Validation/
            blind2_novelty_and_parent_groups.csv
            blind2_family_predictions_with_AD.csv
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


import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. PATHS
# =============================================================================

BASE = data_path('TG')
BLIND2_DIR = data_path('TG/blind2')

TRAIN_PATH = data_path('TG/master_train_features.csv')
WATER_FRACTION_PATH = data_path('TG/original_water_frc.csv')
CORE_GROUP_PATH = data_path('TG/TG_core_nades_groups.csv')

BLIND_FEATURE_PATH = data_path('TG/blind2/master_blind2_features.csv')
BLIND_LABEL_PATH = data_path('TG/blind2/blind2_with_activity.csv')

# These are frozen outputs from the completed aqueous-training blind2 run.
AQUEOUS_OUTPUT_DIR = data_path('TG/blind2/TG_Family_Blind2_Robust_Validation')
BLIND_PARENT_PATH = data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_novelty_and_parent_groups.csv')
AQUEOUS_PREDICTIONS_PATH = data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_predictions_with_AD.csv')

OUTPUT_DIR = data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FROZEN SETTINGS
# =============================================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"
WATER_COLUMN = "water_frc"
CORE_GROUP_COLUMN = "Core_NADES_ID"

# Exact zero is expected. The tolerance only protects against CSV float noise.
PURE_WATER_MAX = 1.0e-10

DESCRIPTOR_BUDGET = 8
PREFILTER_TOP_N = 250
CORRELATION_THRESHOLD = 0.95
MAX_TRAIN_MISSING_FRACTION = 0.95
SEED = 20260803
N_BOOTSTRAP = 5000

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}
RIDGE_ALPHAS = np.logspace(-3.0, 3.0, 13)

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
# 3. INPUT AND FEATURE-SELECTION HELPERS
# =============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


def normalize_code(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def validate_codes(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if CODE_COLUMN not in frame.columns:
        raise ValueError(f"{label}: missing Code column.")
    out = frame.copy()
    out[CODE_COLUMN] = normalize_code(out[CODE_COLUMN])
    if out[CODE_COLUMN].isna().any() or out[CODE_COLUMN].eq("").any():
        raise ValueError(f"{label}: contains missing Code values.")
    if out[CODE_COLUMN].duplicated().any():
        examples = out.loc[out[CODE_COLUMN].duplicated(), CODE_COLUMN].head(10).tolist()
        raise ValueError(f"{label}: duplicated Code values, e.g. {examples}")
    return out


def resolve_column(frame: pd.DataFrame, preferred: str, aliases: tuple[str, ...]) -> str:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in (preferred, *aliases):
        resolved = lookup.get(candidate.casefold())
        if resolved is not None:
            return resolved
    raise ValueError(
        f"Could not find '{preferred}' in columns: {list(frame.columns)}"
    )


def load_feature_table(path: Path, label: str, require_target: bool) -> pd.DataFrame:
    require_file(path, label)
    frame = validate_codes(pd.read_csv(path, low_memory=False), label)
    if require_target:
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(f"{label}: missing {TARGET_COLUMN} column.")
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(f"{label}: target values are missing or non-numeric.")
    elif TARGET_COLUMN in frame.columns:
        raise ValueError(f"{label}: blind feature matrix must not include {TARGET_COLUMN}.")
    return frame


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
    families = {family: [] for family in FAMILY_ORDER}
    for column in columns:
        family = family_of_column(column)
        if family in {
            "Component_A",
            "Component_B",
            "Weighted_mixture",
            "Arithmetic_mean",
            "AB_interactions",
        }:
            families[family].append(column)
    families["Component_A_plus_B"] = (
        families["Component_A"] + families["Component_B"]
    )
    return families


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def clean_train_features(
    train: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    X = train.loc[:, candidate_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    summary = {"input_columns": int(X.shape[1])}

    keep = X.isna().mean(axis=0) <= MAX_TRAIN_MISSING_FRACTION
    summary["removed_over_missing"] = int((~keep).sum())
    X = X.loc[:, keep]

    keep = X.nunique(axis=0, dropna=True) > 1
    summary["removed_constant"] = int((~keep).sum())
    X = X.loc[:, keep]

    duplicate = X.T.duplicated(keep="first")
    summary["removed_duplicate"] = int(duplicate.sum())
    X = X.loc[:, ~duplicate]
    summary["retained_columns"] = int(X.shape[1])
    return X, summary


def select_features_train_only(
    train: pd.DataFrame,
    y: np.ndarray,
    candidate_columns: list[str],
    seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
    """K=8 supervised selection fitted only on the supplied training rows."""
    X_clean, cleaning = clean_train_features(train, candidate_columns)
    if X_clean.shape[1] < DESCRIPTOR_BUDGET:
        raise ValueError(
            f"Only {X_clean.shape[1]} clean descriptors remain; K={DESCRIPTOR_BUDGET}."
        )

    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index,
    )
    y_series = pd.Series(y, index=X.index)
    abs_spearman = X.corrwith(y_series, method="spearman").abs().fillna(0.0)
    screened = abs_spearman.nlargest(
        min(PREFILTER_TOP_N, X.shape[1])
    ).index.tolist()
    X_screen = X.loc[:, screened]
    mi = mutual_info_regression(X_screen.to_numpy(), y, random_state=seed)
    pearson_r2 = X_screen.corrwith(y_series, method="pearson").pow(2).fillna(0.0)

    score = (
        0.50 * percentile_rank(abs_spearman.loc[screened].to_numpy())
        + 0.30 * percentile_rank(mi)
        + 0.20 * percentile_rank(pearson_r2.to_numpy())
    )
    table = pd.DataFrame({
        "feature": screened,
        "abs_spearman_rho": abs_spearman.loc[screened].to_numpy(),
        "mutual_information": mi,
        "univariate_linear_r2": pearson_r2.to_numpy(),
        "prefilter_score": score,
    }).sort_values(
        ["prefilter_score", "feature"], ascending=[False, True]
    ).reset_index(drop=True)

    ordered = table["feature"].tolist()
    correlation = X_screen.loc[:, ordered].corr().abs().fillna(0.0)
    selected: list[str] = []
    for feature in ordered:
        if not selected or (correlation.loc[feature, selected] < CORRELATION_THRESHOLD).all():
            selected.append(feature)
        if len(selected) == DESCRIPTOR_BUDGET:
            break
    if len(selected) != DESCRIPTOR_BUDGET:
        raise ValueError(
            f"Correlation pruning retained {len(selected)}, not K={DESCRIPTOR_BUDGET}."
        )

    ranks = {feature: rank for rank, feature in enumerate(selected, start=1)}
    table["selected"] = table["feature"].isin(selected)
    table["selection_rank"] = table["feature"].map(ranks)
    return selected, table, cleaning


# =============================================================================
# 4. MODELS, AD, AND METRICS
# =============================================================================

def leverage_scores(Z_train: np.ndarray, Z_test: np.ndarray) -> tuple[np.ndarray, float]:
    """Classical leverage in standardized feature space, including intercept."""
    n_train, n_features = Z_train.shape
    X_train = np.column_stack([np.ones(n_train), Z_train])
    X_test = np.column_stack([np.ones(len(Z_test)), Z_test])
    inverse_gram = np.linalg.pinv(X_train.T @ X_train)
    leverage = np.einsum("ij,jk,ik->i", X_test, inverse_gram, X_test)
    h_star = 3.0 * (n_features + 1) / n_train
    return leverage, float(h_star)


def fit_readouts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected: list[str],
    y_train: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    X_train_raw = train.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    X_test_raw = test.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
    X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)

    forest = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    forest.fit(X_train, y_train)

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_test = scaler.transform(X_test)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(Z_train, y_train)

    leverage, h_star = leverage_scores(Z_train, Z_test)
    return (
        {
            "Random forest": forest.predict(X_test),
            "Ridge": ridge.predict(Z_test),
        },
        leverage,
        h_star,
        float(ridge.alpha_),
    )


def metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_mean: float | np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    baseline_prediction = np.asarray(training_mean, dtype=float)
    if baseline_prediction.ndim == 0:
        baseline_prediction = np.full(len(y_true), float(baseline_prediction))
    if len(baseline_prediction) != len(y_true):
        raise ValueError("Baseline prediction and target lengths differ.")
    mse = float(mean_squared_error(y_true, y_pred))
    baseline_mse = float(np.mean((y_true - baseline_prediction) ** 2))
    external_r2 = (
        float(r2_score(y_true, y_pred))
        if len(y_true) >= 2 and np.unique(y_true).size >= 2
        else np.nan
    )
    return {
        "r2_external_mean": external_r2,
        "rmse_K": float(np.sqrt(mse)),
        "mae_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(y_pred - y_true)),
        "train_mean_baseline_rmse_K": float(np.sqrt(baseline_mse)),
        "rmse_over_train_mean_baseline": (
            float(np.sqrt(mse / baseline_mse)) if baseline_mse > 0 else np.nan
        ),
        "mse_skill_vs_train_mean": (
            float(1.0 - mse / baseline_mse) if baseline_mse > 0 else np.nan
        ),
    }


def summarise_predictions(
    predictions: pd.DataFrame,
    grouping: list[str],
    training_mean: float,
) -> pd.DataFrame:
    rows = []
    for keys, subset in predictions.groupby(grouping, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        row["n_systems"] = int(len(subset))
        row["n_parent_groups"] = int(subset["parent_group"].nunique())
        baseline = (
            subset["baseline_train_mean_K"].to_numpy(float)
            if "baseline_train_mean_K" in subset.columns
            else training_mean
        )
        row.update(metric_values(
            subset[TARGET_COLUMN].to_numpy(float),
            subset["predicted_Tg_K"].to_numpy(float),
            baseline,
        ))
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 5. PARENT-GROUP BOOTSTRAP
# =============================================================================

def bootstrap_metrics(
    subset: pd.DataFrame,
    training_mean: float,
    seed: int,
) -> pd.DataFrame:
    subset = subset.reset_index(drop=True)
    groups = subset["parent_group"].astype(str).to_numpy()
    baseline = (
        subset["baseline_train_mean_K"].to_numpy(float)
        if "baseline_train_mean_K" in subset.columns
        else np.full(len(subset), float(training_mean))
    )
    unique_groups = np.unique(groups)
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    unit = (
        "system (= unique parent group)"
        if len(unique_groups) == len(subset)
        else "parent_group"
    )
    rng = np.random.default_rng(seed)
    metric_names = list(metric_values(
        subset[TARGET_COLUMN], subset["predicted_Tg_K"], baseline
    ))
    draws = {name: [] for name in metric_names}

    for _ in range(N_BOOTSTRAP):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([index_by_group[group] for group in chosen])
        values = metric_values(
            subset.loc[indices, TARGET_COLUMN].to_numpy(float),
            subset.loc[indices, "predicted_Tg_K"].to_numpy(float),
            baseline[indices],
        )
        for name, value in values.items():
            if np.isfinite(value):
                draws[name].append(value)

    point = metric_values(
        subset[TARGET_COLUMN].to_numpy(float),
        subset["predicted_Tg_K"].to_numpy(float),
        baseline,
    )
    rows = []
    for metric, values in draws.items():
        values = np.asarray(values, dtype=float)
        rows.append({
            "metric": metric,
            "point_estimate": point[metric],
            "bootstrap_ci_low": np.percentile(values, 2.5) if len(values) else np.nan,
            "bootstrap_ci_high": np.percentile(values, 97.5) if len(values) else np.nan,
            "n_bootstrap_valid": int(len(values)),
            "bootstrap_unit": unit,
            "n_parent_groups": int(len(unique_groups)),
        })
    return pd.DataFrame(rows)


def all_bootstrap_metrics(predictions: pd.DataFrame, training_mean: float) -> pd.DataFrame:
    tables = []
    for number, ((model, family), subset) in enumerate(
        predictions.groupby(["model", "family"], sort=False)
    ):
        table = bootstrap_metrics(subset, training_mean, SEED + 1000 + number)
        table.insert(0, "model", model)
        table.insert(1, "family", family)
        table.insert(2, "family_label", SHORT_LABELS[family])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def pure_vs_aqueous_bootstrap(
    pure: pd.DataFrame,
    aqueous: pd.DataFrame,
) -> pd.DataFrame:
    """Positive delta means the pure-NADES model has lower external error."""
    rows = []
    for number, (key, pure_subset) in enumerate(
        pure.groupby(["model", "family"], sort=False)
    ):
        model, family = key
        aq_subset = aqueous.loc[
            (aqueous["model"] == model) & (aqueous["family"] == family),
            [CODE_COLUMN, "predicted_Tg_K"],
        ].rename(columns={"predicted_Tg_K": "aqueous_prediction"})
        base = pure_subset[[
            CODE_COLUMN, TARGET_COLUMN, "parent_group", "predicted_Tg_K"
        ]].rename(columns={"predicted_Tg_K": "pure_prediction"}).merge(
            aq_subset, on=CODE_COLUMN, how="inner", validate="one_to_one"
        ).reset_index(drop=True)
        if len(base) != len(pure_subset):
            raise ValueError(f"Aqueous predictions do not cover all blind2 codes for {model}/{family}.")

        groups = base["parent_group"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
        unit = (
            "system (= unique parent group)"
            if len(unique_groups) == len(base)
            else "parent_group"
        )
        rng = np.random.default_rng(SEED + 3000 + number)
        draws = {"delta_rmse_K": [], "delta_mae_K": []}
        y = base[TARGET_COLUMN].to_numpy(float)
        pred_pure = base["pure_prediction"].to_numpy(float)
        pred_aq = base["aqueous_prediction"].to_numpy(float)

        for _ in range(N_BOOTSTRAP):
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            index = np.concatenate([index_by_group[group] for group in chosen])
            draws["delta_rmse_K"].append(
                np.sqrt(mean_squared_error(y[index], pred_aq[index]))
                - np.sqrt(mean_squared_error(y[index], pred_pure[index]))
            )
            draws["delta_mae_K"].append(
                mean_absolute_error(y[index], pred_aq[index])
                - mean_absolute_error(y[index], pred_pure[index])
            )

        point = {
            "delta_rmse_K": np.sqrt(mean_squared_error(y, pred_aq)) - np.sqrt(mean_squared_error(y, pred_pure)),
            "delta_mae_K": mean_absolute_error(y, pred_aq) - mean_absolute_error(y, pred_pure),
        }
        for metric, values in draws.items():
            values = np.asarray(values, dtype=float)
            rows.append({
                "model": model,
                "family": family,
                "family_label": SHORT_LABELS[family],
                "comparison": "pure-NADES model vs aqueous-trained model",
                "metric": metric,
                "positive_favors": "pure-NADES model",
                "point_estimate": point[metric],
                "bootstrap_ci_low": np.percentile(values, 2.5),
                "bootstrap_ci_high": np.percentile(values, 97.5),
                "bootstrap_fraction_favoring_pure": float(np.mean(values > 0.0)),
                "n_parent_groups": int(len(unique_groups)),
                "bootstrap_unit": unit,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 6. PURE-NADES TRAIN-ONLY LOGO
# =============================================================================

def run_pure_logo(
    pure: pd.DataFrame,
    family_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logo = LeaveOneGroupOut()
    groups = pure["parent_group"].astype(str).to_numpy()
    y = pure[TARGET_COLUMN].to_numpy(float)

    oof_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    cleaning_rows: list[dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(logo.split(pure, y, groups), start=1):
        train_fold = pure.iloc[train_index].copy()
        test_fold = pure.iloc[test_index].copy()
        y_train = train_fold[TARGET_COLUMN].to_numpy(float)
        held_out_group = "|".join(sorted(test_fold["parent_group"].astype(str).unique()))

        for family_number, family in enumerate(FAMILY_ORDER):
            seed = SEED + fold * 100 + family_number
            selected, table, cleaning = select_features_train_only(
                train_fold, y_train, family_map[family], seed
            )
            readouts, leverage, h_star, ridge_alpha = fit_readouts(
                train_fold, test_fold, selected, y_train, seed
            )

            table.insert(0, "fold", fold)
            table.insert(1, "held_out_parent_group", held_out_group)
            table.insert(2, "family", family)
            table.insert(3, "family_label", SHORT_LABELS[family])
            selected_rows.append(table)
            cleaning_rows.append({
                "fold": fold,
                "held_out_parent_group": held_out_group,
                "family": family,
                "family_label": SHORT_LABELS[family],
                "n_selected": len(selected),
                "ridge_alpha_train_only": ridge_alpha,
                **cleaning,
            })

            for model, prediction in readouts.items():
                oof_rows.append(pd.DataFrame({
                    CODE_COLUMN: test_fold[CODE_COLUMN].to_numpy(),
                    TARGET_COLUMN: test_fold[TARGET_COLUMN].to_numpy(float),
                    "parent_group": test_fold["parent_group"].to_numpy(),
                    "fold": fold,
                    "model": model,
                    "family": family,
                    "family_label": SHORT_LABELS[family],
                    "n_selected_descriptors": len(selected),
                    "predicted_Tg_K": prediction,
                    "baseline_train_mean_K": float(np.mean(y_train)),
                    "leverage": leverage,
                    "h_star": h_star,
                    "within_leverage_AD": leverage <= h_star,
                    "ridge_alpha_train_only": ridge_alpha,
                }))

        print(f"LOGO fold {fold:02d}/{pure['parent_group'].nunique():02d} | held-out {held_out_group}")

    return (
        pd.concat(oof_rows, ignore_index=True),
        pd.concat(selected_rows, ignore_index=True),
        pd.DataFrame(cleaning_rows),
    )


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    required = [
        (TRAIN_PATH, "labelled Tg feature matrix"),
        (WATER_FRACTION_PATH, "water-fraction table"),
        (CORE_GROUP_PATH, "Core NADES group table"),
        (BLIND_FEATURE_PATH, "blind2 feature matrix"),
        (BLIND_LABEL_PATH, "blind2 labels"),
        (BLIND_PARENT_PATH, "blind2 parent-group metadata from the frozen aqueous run"),
        (AQUEOUS_PREDICTIONS_PATH, "frozen aqueous-model blind2 predictions"),
    ]
    for path, label in required:
        require_file(path, label)

    train = load_feature_table(TRAIN_PATH, "labelled Tg feature matrix", require_target=True)
    blind = load_feature_table(BLIND_FEATURE_PATH, "blind2 feature matrix", require_target=False)

    water = validate_codes(pd.read_csv(WATER_FRACTION_PATH, low_memory=False), WATER_FRACTION_PATH.name)
    water_column = resolve_column(water, WATER_COLUMN, ("water_fraction", "water_frac", "H2O_fraction"))
    water = water[[CODE_COLUMN, water_column]].rename(columns={water_column: WATER_COLUMN})
    water[WATER_COLUMN] = pd.to_numeric(water[WATER_COLUMN], errors="coerce")
    if water[WATER_COLUMN].isna().any() or (water[WATER_COLUMN] < 0).any() or (water[WATER_COLUMN] > 1).any():
        raise ValueError("water_frc must be numeric and lie between 0 and 1.")

    core = validate_codes(pd.read_csv(CORE_GROUP_PATH, low_memory=False), CORE_GROUP_PATH.name)
    core_column = resolve_column(core, CORE_GROUP_COLUMN, ("core_nades_id", "parent_group", "CoreNADESID"))
    core = core[[CODE_COLUMN, core_column]].rename(columns={core_column: "parent_group"})
    core["parent_group"] = core["parent_group"].astype("string").str.strip()
    if core["parent_group"].isna().any() or core["parent_group"].eq("").any():
        raise ValueError("Core NADES group table contains missing group identifiers.")

    train = train.merge(water, on=CODE_COLUMN, how="left", validate="one_to_one")
    train = train.merge(core, on=CODE_COLUMN, how="left", validate="one_to_one")
    if train[[WATER_COLUMN, "parent_group"]].isna().any().any():
        missing = train.loc[train[[WATER_COLUMN, "parent_group"]].isna().any(axis=1), CODE_COLUMN].head(10).tolist()
        raise ValueError(f"Training metadata missing for Codes such as {missing}")

    pure = train.loc[train[WATER_COLUMN] <= PURE_WATER_MAX].copy().reset_index(drop=True)
    if len(pure) < 12 or pure["parent_group"].nunique() < 10:
        raise ValueError(
            f"Pure-NADES subset is too small: {len(pure)} rows / "
            f"{pure['parent_group'].nunique()} parent groups."
        )

    family_map = build_family_map(train.columns)
    expected = set().union(*family_map.values())
    missing = sorted(expected - set(blind.columns))
    if missing:
        raise ValueError(f"blind2 lacks {len(missing)} descriptors, e.g. {missing[:12]}")

    blind_parent = validate_codes(pd.read_csv(BLIND_PARENT_PATH, low_memory=False), BLIND_PARENT_PATH.name)
    required_parent_columns = {CODE_COLUMN, "parent_group", "novelty_group"}
    if not required_parent_columns.issubset(blind_parent.columns):
        raise ValueError(f"{BLIND_PARENT_PATH.name} must contain {required_parent_columns}.")
    blind_parent = blind_parent[[CODE_COLUMN, "parent_group", "novelty_group"]]
    blind = blind.merge(blind_parent, on=CODE_COLUMN, how="left", validate="one_to_one")
    if blind[["parent_group", "novelty_group"]].isna().any().any():
        raise ValueError("blind2 parent/novelty metadata are incomplete.")

    print("=" * 105)
    print("TG PURE-NADES CONTROL — TRAIN-ONLY LOGO + POST-HOC BLIND2 DIAGNOSTIC")
    print("=" * 105)
    print(f"All labelled Tg rows          : {len(train)}")
    print(f"Pure NADES rows (water=0)    : {len(pure)}")
    print(f"Pure NADES parent groups     : {pure['parent_group'].nunique()}")
    print(f"Pure water threshold         : {PURE_WATER_MAX:g}")
    print(f"Descriptor budget            : {DESCRIPTOR_BUDGET} per family")
    print(f"Internal validation          : LeaveOneGroupOut by Core_NADES_ID")
    print(f"blind2 systems / parents     : {len(blind)} / {blind['parent_group'].nunique()}")
    print("No water rescaling is applied: pure rows already have multiplier (1-water_frc)=1.")
    print("blind2 is post-hoc diagnostic only; it must not trigger parameter changes.\n")

    # -------------------------------------------------------------------------
    # A. Honest pure-NADES internal validation: all selection inside LOGO train.
    # -------------------------------------------------------------------------
    oof, logo_selected, logo_cleaning = run_pure_logo(pure, family_map)
    pure_training_mean = float(pure[TARGET_COLUMN].mean())
    oof_metrics = summarise_predictions(
        oof, ["model", "family", "family_label"], pure_training_mean
    )
    oof_bootstrap = all_bootstrap_metrics(oof, pure_training_mean)

    oof.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_logo_oof_predictions.csv'), index=False)
    oof_metrics.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_logo_pooled_metrics.csv'), index=False)
    oof_bootstrap.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_logo_bootstrap_intervals.csv'), index=False)
    logo_selected.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_logo_selected_features.csv'), index=False)
    logo_cleaning.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_logo_cleaning_log.csv'), index=False)

    # -------------------------------------------------------------------------
    # B. Frozen pure-NADES models fit on ALL pure-labelled rows.
    # blind2 labels are deliberately not loaded in this section.
    # -------------------------------------------------------------------------
    blind_without_labels = blind.drop(columns=[column for column in [TARGET_COLUMN] if column in blind.columns])
    final_predictions: list[pd.DataFrame] = []
    final_selected: list[pd.DataFrame] = []
    final_cleaning: list[dict[str, object]] = []
    y_pure = pure[TARGET_COLUMN].to_numpy(float)

    for family_number, family in enumerate(FAMILY_ORDER):
        seed = SEED + 9000 + family_number
        selected, table, cleaning = select_features_train_only(
            pure, y_pure, family_map[family], seed
        )
        readouts, leverage, h_star, ridge_alpha = fit_readouts(
            pure, blind_without_labels, selected, y_pure, seed
        )
        table.insert(0, "family", family)
        table.insert(1, "family_label", SHORT_LABELS[family])
        final_selected.append(table)
        final_cleaning.append({
            "family": family,
            "family_label": SHORT_LABELS[family],
            "n_selected": len(selected),
            "ridge_alpha_train_only": ridge_alpha,
            **cleaning,
        })
        for model, prediction in readouts.items():
            final_predictions.append(pd.DataFrame({
                CODE_COLUMN: blind[CODE_COLUMN].to_numpy(),
                "parent_group": blind["parent_group"].to_numpy(),
                "novelty_group": blind["novelty_group"].to_numpy(),
                "model": model,
                "family": family,
                "family_label": SHORT_LABELS[family],
                "n_selected_descriptors": len(selected),
                "predicted_Tg_K": prediction,
                "baseline_train_mean_K": pure_training_mean,
                "leverage": leverage,
                "h_star": h_star,
                "within_leverage_AD": leverage <= h_star,
                "ridge_alpha_train_only": ridge_alpha,
            }))
        print(
            f"Frozen pure model | {SHORT_LABELS[family]:22s} | "
            f"K={len(selected)} | Ridge alpha={ridge_alpha:.4g} | "
            f"outside AD={int((leverage > h_star).sum())}/{len(leverage)}"
        )

    pure_blind = pd.concat(final_predictions, ignore_index=True)
    pd.concat(final_selected, ignore_index=True).to_csv(
        data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_final_selected_features.csv'), index=False
    )
    pd.DataFrame(final_cleaning).to_csv(
        data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_final_cleaning_log.csv'), index=False
    )

    # -------------------------------------------------------------------------
    # C. blind2 labels enter ONLY after all pure-model predictions are frozen.
    # -------------------------------------------------------------------------
    labels = load_feature_table(BLIND_LABEL_PATH, "blind2 Tg labels", require_target=True)
    labels = labels[[CODE_COLUMN, TARGET_COLUMN]]
    if set(blind[CODE_COLUMN]) != set(labels[CODE_COLUMN]):
        raise ValueError("blind2 feature and label Code sets do not match exactly.")
    pure_blind = pure_blind.merge(labels, on=CODE_COLUMN, how="left", validate="many_to_one")
    pure_blind["error_K"] = pure_blind["predicted_Tg_K"] - pure_blind[TARGET_COLUMN]
    pure_blind["absolute_error_K"] = pure_blind["error_K"].abs()

    blind_metrics = summarise_predictions(
        pure_blind, ["model", "family", "family_label"], pure_training_mean
    )
    blind_metrics_by_novelty = summarise_predictions(
        pure_blind, ["model", "family", "family_label", "novelty_group"], pure_training_mean
    )
    blind_metrics_by_ad = summarise_predictions(
        pure_blind, ["model", "family", "family_label", "within_leverage_AD"], pure_training_mean
    )
    blind_bootstrap = all_bootstrap_metrics(pure_blind, pure_training_mean)

    pure_blind.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_blind2_predictions_with_AD.csv'), index=False)
    blind_metrics.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_blind2_metrics.csv'), index=False)
    blind_metrics_by_novelty.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_blind2_metrics_by_novelty.csv'), index=False)
    blind_metrics_by_ad.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_blind2_metrics_by_AD.csv'), index=False)
    blind_bootstrap.to_csv(data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_blind2_bootstrap_intervals.csv'), index=False)

    # -------------------------------------------------------------------------
    # D. Direct diagnostic contrast against the already frozen aqueous model.
    # A positive delta means pure-only training reduced external error.
    # -------------------------------------------------------------------------
    aqueous = pd.read_csv(AQUEOUS_PREDICTIONS_PATH, low_memory=False)
    if CODE_COLUMN not in aqueous.columns:
        raise ValueError(f"{AQUEOUS_PREDICTIONS_PATH.name}: missing Code column.")
    aqueous[CODE_COLUMN] = normalize_code(aqueous[CODE_COLUMN])
    if aqueous[CODE_COLUMN].isna().any() or aqueous[CODE_COLUMN].eq("").any():
        raise ValueError(f"{AQUEOUS_PREDICTIONS_PATH.name}: missing Code values.")
    required_aqueous = {CODE_COLUMN, "model", "family", "predicted_Tg_K"}
    if not required_aqueous.issubset(aqueous.columns):
        raise ValueError(f"Aqueous prediction file must contain {required_aqueous}.")
    aqueous = aqueous[[CODE_COLUMN, "model", "family", "predicted_Tg_K"]]
    if aqueous.duplicated([CODE_COLUMN, "model", "family"]).any():
        raise ValueError(
            "Aqueous prediction file has duplicate Code/model/family rows."
        )
    comparison = pure_vs_aqueous_bootstrap(pure_blind, aqueous)
    comparison.to_csv(
        data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/blind2_pure_vs_aqueous_paired_bootstrap_contrasts.csv'),
        index=False,
    )

    print("\n" + "=" * 105)
    print("PURE-NADES LOGO: POOLED INTERNAL RESULTS")
    print("=" * 105)
    print(oof_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n" + "=" * 105)
    print("PURE-NADES MODEL ON BLIND2 — POST-HOC DIAGNOSTIC ONLY")
    print("=" * 105)
    print(blind_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n" + "=" * 105)
    print("PURE-NADES MINUS AQUEOUS-TRAINED MODEL: PAIRED BLIND2 CONTRAST")
    print("Positive delta = pure-only model has lower error.")
    print("=" * 105)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved to:", OUTPUT_DIR)
    print("\nInterpretation rule: blind2 was already opened. These contrasts test the water-domain")
    print("hypothesis but cannot be reported as a newly independent external validation.")


if __name__ == "__main__":
    main()


