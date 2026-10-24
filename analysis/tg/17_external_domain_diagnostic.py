

"""
TG FAMILY READOUT — ROBUST BLIND2 VALIDATION
=============================================

Controlled validation of descriptor families on blind2.  This extends the
earlier K=8 family-readout script with:

1. paired parent-group bootstrap uncertainty intervals (equivalent to a
   system bootstrap only when every blind system has a unique parent);
2. train-only leverage applicability-domain (AD) diagnostics;
3. RidgeCV readout sensitivity control on the SAME selected descriptors,
   with alpha selected only from labelled training data;
4. novelty stratification: new components vs known-component novel formulations;
5. RMSE/MAE and a frozen training-mean baseline as primary external metrics.

This is a representation experiment. It does not replace the Final60
deployment model or permit further selection of K, descriptor families, or
algorithm after blind2 evaluation.

Required files
--------------
TG/
    master_train_features.csv
    original.csv                         # Code, A_id...D_id, activity
    a.csv, b.csv, c.csv, d.csv           # Code and val1...val4
    blind2/
        master_blind2_features.csv
        blind2.csv                       # Code, A_id...D_id
        blind2_with_activity.csv         # Code, A_id...D_id, activity
        a.csv, b.csv, c.csv, d.csv       # Code and val1...val4
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
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. PATHS
# =============================================================================

BASE = data_path('TG')
BLIND2_DIR = data_path('TG/blind2')

TRAIN_FEATURE_PATH = data_path('TG/master_train_features.csv')
TRAIN_METADATA_PATH = data_path('TG/original.csv')
BLIND_FEATURE_PATH = data_path('TG/blind2/master_blind2_features.csv')
BLIND_METADATA_PATH = data_path('TG/blind2/blind2.csv')
BLIND_LABEL_PATH = data_path('TG/blind2/blind2_with_activity.csv')

TRAIN_COMPONENT_PATHS = {
    "A": data_path('TG/a.csv'),
    "B": data_path('TG/b.csv'),
    "C": data_path('TG/c.csv'),
    "D": data_path('TG/d.csv'),
}
BLIND_COMPONENT_PATHS = {
    "A": data_path('TG/blind2/a.csv'),
    "B": data_path('TG/blind2/b.csv'),
    "C": data_path('TG/blind2/c.csv'),
    "D": data_path('TG/blind2/d.csv'),
}

OUTPUT_DIR = data_path('TG/blind2/TG_Family_Blind2_Robust_Validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FROZEN FAMILY-READOUT SETTINGS
# =============================================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"
COMPONENTS = ("A", "B", "C", "D")

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

# Ridge regularization is selected only from the labelled training data.
# It never sees blind2 Tg values.  This makes Ridge a useful readout check
# without arbitrarily handicapping one descriptor family.
RIDGE_ALPHAS = np.logspace(-3.0, 3.0, 13)
N_BOOTSTRAP = 5000

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

FAMILY_COMPARISONS = [
    ("Weighted_mixture", "Arithmetic_mean"),
    ("Weighted_mixture", "Component_A"),
    ("Weighted_mixture", "Component_B"),
    ("Weighted_mixture", "Component_A_plus_B"),
    ("Weighted_mixture", "AB_interactions"),
]


# =============================================================================
# 3. BASIC DATA HELPERS
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


def normalize_component(value: object) -> str | None:
    if pd.isna(value):
        return None
    token = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    return token or None


def validate_codes(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if CODE_COLUMN not in frame.columns:
        raise ValueError(f"{label}: missing Code column.")
    out = frame.copy()
    out[CODE_COLUMN] = normalize_code(out[CODE_COLUMN])
    if out[CODE_COLUMN].isna().any() or out[CODE_COLUMN].eq("").any():
        raise ValueError(f"{label}: contains missing Code values.")
    if out[CODE_COLUMN].duplicated().any():
        examples = out.loc[out[CODE_COLUMN].duplicated(), CODE_COLUMN].head(10).tolist()
        raise ValueError(f"{label}: duplicated Code values: {examples}")
    return out


def load_feature_table(path: Path, label: str, needs_activity: bool) -> pd.DataFrame:
    require_file(path, label)
    frame = validate_codes(pd.read_csv(path, low_memory=False), label)
    if needs_activity:
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(f"{label}: missing activity column.")
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        if frame[TARGET_COLUMN].isna().any():
            raise ValueError(f"{label}: activity contains missing/non-numeric values.")
    elif TARGET_COLUMN in frame.columns:
        raise ValueError(f"{label}: a blind feature file must not contain activity.")
    return frame


# =============================================================================
# 4. FAMILY MAP AND TRAIN-ONLY SELECTION
# =============================================================================

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


def clean_train_features(train: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
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
    """Exact K=8 regression prefilter used for the prior family readouts."""
    X_clean, log = clean_train_features(train, candidate_columns)
    if X_clean.shape[1] < DESCRIPTOR_BUDGET:
        raise ValueError(f"Only {X_clean.shape[1]} clean columns remain; K={DESCRIPTOR_BUDGET}.")

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

    selection_rank = {feature: rank for rank, feature in enumerate(selected, start=1)}
    scores["selected"] = scores["feature"].isin(selected)
    scores["selection_rank"] = scores["feature"].map(selection_rank)
    return selected, scores, log


# =============================================================================
# 5. COMPONENT NOVELTY AND BOOTSTRAP PARENT GROUPS
# =============================================================================

def load_coefficients(paths: dict[str, Path], label: str) -> pd.DataFrame:
    result: pd.DataFrame | None = None
    for position, path in paths.items():
        require_file(path, f"{label} {position}-component table")
        value_column = f"val{COMPONENTS.index(position) + 1}"
        frame = pd.read_csv(path, low_memory=False)
        if CODE_COLUMN not in frame.columns:
            raise ValueError(f"{path.name}: missing Code column.")
        if value_column not in frame.columns:
            raise ValueError(f"{path.name}: missing {value_column}.")
        # Some optional-component CSVs contain trailing completely empty rows.
        # They carry no mixture information and are safely removed.  A blank
        # Code paired with an actual coefficient remains an input error.
        normalized_codes = normalize_code(frame[CODE_COLUMN])
        missing_code = normalized_codes.isna() | normalized_codes.eq("")
        values_before_drop = pd.to_numeric(frame[value_column], errors="coerce")
        invalid_rows = missing_code & values_before_drop.notna() & values_before_drop.ne(0.0)
        if invalid_rows.any():
            examples = frame.index[invalid_rows].tolist()[:10]
            raise ValueError(
                f"{path.name}: Code is missing for non-zero {value_column} rows "
                f"at indices {examples}."
            )
        frame = frame.loc[~missing_code].copy()
        frame[CODE_COLUMN] = normalize_code(frame[CODE_COLUMN])
        frame = validate_codes(frame, path.name)
        frame = frame[[CODE_COLUMN, value_column]].copy()
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        if (frame[value_column] < 0).any():
            raise ValueError(f"{path.name}: {value_column} contains a negative coefficient.")
        # The original encoder interprets an omitted optional C/D coefficient
        # as an absent component, i.e. coefficient zero.
        frame[value_column] = frame[value_column].fillna(0.0)
        frame = frame.rename(columns={value_column: f"{position}_val"})
        result = frame if result is None else result.merge(frame, on=CODE_COLUMN, how="outer", validate="one_to_one")
    assert result is not None
    # A missing C/D row means that the optional component is absent, exactly as
    # in the original mixture encoder. It is therefore a zero coefficient, not
    # a missing experimental value.
    for component in COMPONENTS:
        column = f"{component}_val"
        if column not in result.columns:
            result[column] = 0.0
        result[column] = result[column].fillna(0.0)
    return result


def load_metadata(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    frame = validate_codes(pd.read_csv(path, low_memory=False), label)
    required = [f"{component}_id" for component in COMPONENTS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}: missing component-identity columns {missing}.")
    return frame[[CODE_COLUMN, *required]].copy()


def formulation_metadata(
    metadata: pd.DataFrame,
    coefficients: pd.DataFrame,
    known_components: set[str] | None = None,
    seen_formulations: set[str] | None = None,
) -> pd.DataFrame:
    frame = metadata.merge(coefficients, on=CODE_COLUMN, how="left", validate="one_to_one")
    rows = []
    for _, row in frame.iterrows():
        pairs: list[tuple[str, float]] = []
        for component in COMPONENTS:
            token = normalize_component(row[f"{component}_id"])
            coefficient = float(row[f"{component}_val"])
            if token is not None and coefficient > 0:
                pairs.append((token, coefficient))
            elif token is None and coefficient > 0:
                raise ValueError(f"{row[CODE_COLUMN]}: positive {component} coefficient but missing {component}_id.")
        if len(pairs) < 2:
            raise ValueError(f"{row[CODE_COLUMN]}: fewer than two present components.")
        total = sum(value for _, value in pairs)
        parent_signature = "|".join(sorted(token for token, _ in pairs))
        formulation_signature = "|".join(
            sorted(f"{token}@{value / total:.8f}" for token, value in pairs)
        )
        contains_new = (
            False if known_components is None else any(token not in known_components for token, _ in pairs)
        )
        if known_components is None:
            novelty = "training_reference"
        elif contains_new:
            novelty = "contains_new_component"
        elif seen_formulations is not None and formulation_signature in seen_formulations:
            novelty = "seen_formulation"
        else:
            novelty = "known_components_new_formulation_or_ratio"
        rows.append({
            CODE_COLUMN: row[CODE_COLUMN],
            "parent_group": parent_signature,
            "formulation_signature": formulation_signature,
            "contains_new_component": contains_new,
            "novelty_group": novelty,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. MODELS, LEVERAGE AD, AND EXTERNAL METRICS
# =============================================================================

def calculate_leverage(Z_train: np.ndarray, Z_blind: np.ndarray) -> tuple[np.ndarray, float]:
    """Linear leverage diagnostic in the train-only standardized K=8 space."""
    n_train, n_features = Z_train.shape
    train_design = np.column_stack([np.ones(n_train), Z_train])
    blind_design = np.column_stack([np.ones(len(Z_blind)), Z_blind])
    precision = np.linalg.pinv(train_design.T @ train_design)
    leverage = np.einsum("ij,jk,ik->i", blind_design, precision, blind_design)
    h_star = 3.0 * (n_features + 1) / n_train
    return leverage, h_star


def fit_and_predict_readouts(
    train: pd.DataFrame,
    blind: pd.DataFrame,
    selected_features: list[str],
    y_train: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, float, float]:
    X_train = train.loc[:, selected_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_blind = blind.loc[:, selected_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_blind = imputer.transform(X_blind)

    rf = RandomForestRegressor(random_state=seed, **RF_PARAMS)
    rf.fit(X_train, y_train)

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_blind = scaler.transform(X_blind)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(Z_train, y_train)

    leverage, h_star = calculate_leverage(Z_train, Z_blind)
    return (
        {"Random forest": rf.predict(X_blind), "Ridge": ridge.predict(Z_blind)},
        leverage,
        h_star,
        float(ridge.alpha_),
    )


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    baseline_mse = float(np.mean((y_true - train_mean) ** 2))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and np.unique(y_true).size >= 2 else np.nan
    return {
        "r2_external_mean": r2,
        "rmse_K": float(np.sqrt(mse)),
        "mae_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(y_pred - y_true)),
        "train_mean_baseline_rmse_K": float(np.sqrt(baseline_mse)),
        "rmse_over_train_mean_baseline": float(np.sqrt(mse / baseline_mse)) if baseline_mse > 0 else np.nan,
        "mse_skill_vs_train_mean": float(1.0 - mse / baseline_mse) if baseline_mse > 0 else np.nan,
    }


def summarise_groups(
    frame: pd.DataFrame,
    grouping: list[str],
    train_mean: float,
) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(grouping, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(grouping, keys))
        record["n_systems"] = len(subset)
        record["n_parent_groups"] = subset["parent_group"].nunique()
        record.update(metric_values(subset[TARGET_COLUMN], subset["predicted_Tg_K"], train_mean))
        rows.append(record)
    return pd.DataFrame(rows)


# =============================================================================
# 7. GROUP BOOTSTRAP AND PAIRED FAMILY CONTRASTS
# =============================================================================

def bootstrap_summary(
    frame: pd.DataFrame,
    train_mean: float,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap resamples parent groups; uses Code when every parent is unique."""
    frame = frame.reset_index(drop=True)
    groups = frame["parent_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    bootstrap_unit = (
        "system (= unique parent group)"
        if len(unique_groups) == len(frame)
        else "parent_group"
    )
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    metric_names = list(metric_values(frame[TARGET_COLUMN], frame["predicted_Tg_K"], train_mean))
    samples = {metric: [] for metric in metric_names}

    for _ in range(N_BOOTSTRAP):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_index = np.concatenate([index_by_group[group] for group in sampled_groups])
        values = metric_values(
            frame.loc[sampled_index, TARGET_COLUMN].to_numpy(),
            frame.loc[sampled_index, "predicted_Tg_K"].to_numpy(),
            train_mean,
        )
        for metric, value in values.items():
            if np.isfinite(value):
                samples[metric].append(value)

    point = metric_values(frame[TARGET_COLUMN], frame["predicted_Tg_K"], train_mean)
    rows = []
    for metric, values in samples.items():
        values = np.asarray(values, dtype=float)
        rows.append({
            "metric": metric,
            "point_estimate": point[metric],
            "bootstrap_ci_low": np.percentile(values, 2.5) if len(values) else np.nan,
            "bootstrap_ci_high": np.percentile(values, 97.5) if len(values) else np.nan,
            "n_bootstrap_valid": len(values),
            "bootstrap_unit": bootstrap_unit,
            "n_parent_groups": len(unique_groups),
        })
    return pd.DataFrame(rows)


def all_bootstrap_summaries(predictions: pd.DataFrame, train_mean: float) -> pd.DataFrame:
    tables = []
    for number, ((model, family), subset) in enumerate(predictions.groupby(["model", "family"], sort=False)):
        table = bootstrap_summary(subset, train_mean, SEED + 1000 + number)
        table.insert(0, "model", model)
        table.insert(1, "family", family)
        table.insert(2, "family_label", SHORT_LABELS[family])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def paired_bootstrap_contrasts(predictions: pd.DataFrame, train_mean: float) -> pd.DataFrame:
    rows = []
    for model_number, model in enumerate(predictions["model"].unique()):
        model_frame = predictions.loc[predictions["model"] == model].copy()
        base = model_frame[[CODE_COLUMN, TARGET_COLUMN, "parent_group"]].drop_duplicates(CODE_COLUMN)
        groups = base["parent_group"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        bootstrap_unit = (
            "system (= unique parent group)"
            if len(unique_groups) == len(base)
            else "parent_group"
        )
        index_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
        rng = np.random.default_rng(SEED + 2000 + model_number)
        pivot = model_frame.pivot(index=CODE_COLUMN, columns="family", values="predicted_Tg_K").reindex(base[CODE_COLUMN])
        y = base[TARGET_COLUMN].to_numpy(float)

        for first, second in FAMILY_COMPARISONS:
            if first not in pivot or second not in pivot:
                continue
            left, right = pivot[first].to_numpy(float), pivot[second].to_numpy(float)
            point_left, point_right = metric_values(y, left, train_mean), metric_values(y, right, train_mean)
            values = {"delta_rmse_K": [], "delta_mae_K": [], "delta_skill": []}
            for _ in range(N_BOOTSTRAP):
                sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
                idx = np.concatenate([index_by_group[group] for group in sampled_groups])
                left_metrics, right_metrics = metric_values(y[idx], left[idx], train_mean), metric_values(y[idx], right[idx], train_mean)
                values["delta_rmse_K"].append(right_metrics["rmse_K"] - left_metrics["rmse_K"])
                values["delta_mae_K"].append(right_metrics["mae_K"] - left_metrics["mae_K"])
                values["delta_skill"].append(left_metrics["mse_skill_vs_train_mean"] - right_metrics["mse_skill_vs_train_mean"])
            point = {
                "delta_rmse_K": point_right["rmse_K"] - point_left["rmse_K"],
                "delta_mae_K": point_right["mae_K"] - point_left["mae_K"],
                "delta_skill": point_left["mse_skill_vs_train_mean"] - point_right["mse_skill_vs_train_mean"],
            }
            for metric, bootstrap_values in values.items():
                bootstrap_values = np.asarray(bootstrap_values, dtype=float)
                rows.append({
                    "model": model,
                    "first_family": first,
                    "second_family": second,
                    "comparison": f"{SHORT_LABELS[first]} vs {SHORT_LABELS[second]}",
                    "metric": metric,
                    "positive_favors": first,
                    "point_estimate": point[metric],
                    "bootstrap_ci_low": np.percentile(bootstrap_values, 2.5),
                    "bootstrap_ci_high": np.percentile(bootstrap_values, 97.5),
                    "bootstrap_fraction_favoring_first": float(np.mean(bootstrap_values > 0.0)),
                    "n_parent_groups": len(unique_groups),
                    "bootstrap_unit": bootstrap_unit,
                })
    return pd.DataFrame(rows)


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:
    required = [
        (TRAIN_FEATURE_PATH, "Tg training feature matrix"),
        (TRAIN_METADATA_PATH, "training component metadata"),
        (BLIND_FEATURE_PATH, "blind2 feature matrix"),
        (BLIND_METADATA_PATH, "blind2 component metadata"),
        (BLIND_LABEL_PATH, "blind2 Tg labels"),
        *[(path, f"training {position}.csv") for position, path in TRAIN_COMPONENT_PATHS.items()],
        *[(path, f"blind2 {position}.csv") for position, path in BLIND_COMPONENT_PATHS.items()],
    ]
    for path, label in required:
        require_file(path, label)

    train = load_feature_table(TRAIN_FEATURE_PATH, "Tg training feature matrix", needs_activity=True)
    blind = load_feature_table(BLIND_FEATURE_PATH, "blind2 feature matrix", needs_activity=False)
    blind_labels = load_feature_table(BLIND_LABEL_PATH, "blind2 Tg labels", needs_activity=True)[[CODE_COLUMN, TARGET_COLUMN]]
    if set(blind[CODE_COLUMN]) != set(blind_labels[CODE_COLUMN]):
        raise ValueError("blind2 feature and label Code sets do not match exactly.")

    y_train = train[TARGET_COLUMN].to_numpy(float)
    train_mean = float(np.mean(y_train))
    family_map = build_family_map(train.columns)
    expected_feature_columns = set().union(*family_map.values())
    missing = sorted(expected_feature_columns - set(blind.columns))
    if missing:
        raise ValueError(f"blind2 feature matrix lacks {len(missing)} descriptor columns, e.g. {missing[:15]}")

    train_metadata = load_metadata(TRAIN_METADATA_PATH, "original.csv")
    blind_metadata = load_metadata(BLIND_METADATA_PATH, "blind2.csv")
    train_coefficients = load_coefficients(TRAIN_COMPONENT_PATHS, "training")
    blind_coefficients = load_coefficients(BLIND_COMPONENT_PATHS, "blind2")
    train_formulations = formulation_metadata(train_metadata, train_coefficients)
    known_components = set()
    for column in [f"{component}_id" for component in COMPONENTS]:
        known_components.update(
            token for token in train_metadata[column].map(normalize_component) if token is not None
        )
    blind_formulations = formulation_metadata(
        blind_metadata,
        blind_coefficients,
        known_components=known_components,
        seen_formulations=set(train_formulations["formulation_signature"]),
    )
    blind = blind.merge(blind_formulations, on=CODE_COLUMN, how="left", validate="one_to_one")
    blind = blind.merge(blind_labels, on=CODE_COLUMN, how="left", validate="one_to_one")

    print("=" * 105)
    print("TG FAMILY READOUT — ROBUST BLIND2 VALIDATION")
    print("=" * 105)
    print(f"Labelled training rows     : {len(train)}")
    print(f"Blind2 systems             : {len(blind)}")
    print(f"Blind2 parent groups       : {blind['parent_group'].nunique()}")
    print(f"Descriptor budget          : {DESCRIPTOR_BUDGET} per family")
    print(f"Bootstrap                  : {N_BOOTSTRAP} parent-group resamples")
    print("Ridge alpha                : selected by RidgeCV on labelled training data only")
    print("Blind2 Tg values enter only after all RF and Ridge prediction vectors are created.\n")
    print("Blind2 novelty composition:")
    print(blind["novelty_group"].value_counts().to_string())

    prediction_tables, selection_tables, cleaning_rows = [], [], []
    blind_without_y = blind.drop(columns=[TARGET_COLUMN])
    for family_number, family in enumerate(FAMILY_ORDER):
        selected, scores, cleaning = select_features_train_only(
            train, y_train, family_map[family], SEED + family_number
        )
        predictions, leverage, h_star, ridge_alpha = fit_and_predict_readouts(
            train, blind_without_y, selected, y_train, SEED + family_number
        )
        for model_name, values in predictions.items():
            prediction_tables.append(pd.DataFrame({
                CODE_COLUMN: blind[CODE_COLUMN].to_numpy(),
                "model": model_name,
                "family": family,
                "family_label": SHORT_LABELS[family],
                "n_selected_descriptors": len(selected),
                TARGET_COLUMN: blind[TARGET_COLUMN].to_numpy(float),
                "predicted_Tg_K": values,
                "parent_group": blind["parent_group"].to_numpy(),
                "formulation_signature": blind["formulation_signature"].to_numpy(),
                "novelty_group": blind["novelty_group"].to_numpy(),
                "contains_new_component": blind["contains_new_component"].to_numpy(),
                "leverage": leverage,
                "h_star": h_star,
                "within_leverage_AD": leverage <= h_star,
                "ridge_alpha_train_only": ridge_alpha,
            }))
        scores.insert(0, "family", family)
        scores.insert(1, "family_label", SHORT_LABELS[family])
        selection_tables.append(scores)
        cleaning_rows.append({"family": family, "family_label": SHORT_LABELS[family], "n_selected": len(selected), **cleaning})
        print(
            f"{SHORT_LABELS[family]:22s}: K={len(selected)}; "
            f"RF + Ridge predictions complete (Ridge alpha={ridge_alpha:.4g})"
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions["error_K"] = predictions["predicted_Tg_K"] - predictions[TARGET_COLUMN]
    predictions["absolute_error_K"] = predictions["error_K"].abs()

    full_metrics = summarise_groups(predictions, ["model", "family", "family_label"], train_mean)
    novelty_metrics = summarise_groups(predictions, ["model", "family", "family_label", "novelty_group"], train_mean)
    ad_metrics = summarise_groups(predictions, ["model", "family", "family_label", "within_leverage_AD"], train_mean)
    bootstrap_metrics = all_bootstrap_summaries(predictions, train_mean)
    paired_contrasts = paired_bootstrap_contrasts(predictions, train_mean)

    predictions.to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_predictions_with_AD.csv'), index=False)
    full_metrics.to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_external_metrics.csv'), index=False)
    novelty_metrics.to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_metrics_by_novelty.csv'), index=False)
    ad_metrics.to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_metrics_by_AD.csv'), index=False)
    bootstrap_metrics.to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_bootstrap_intervals.csv'), index=False)
    paired_contrasts.to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_paired_family_bootstrap_contrasts.csv'), index=False)
    pd.concat(selection_tables, ignore_index=True).to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_selected_features.csv'), index=False)
    pd.DataFrame(cleaning_rows).to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_family_cleaning_log.csv'), index=False)
    blind_formulations.to_csv(data_path('TG/blind2/TG_Family_Blind2_Robust_Validation/blind2_novelty_and_parent_groups.csv'), index=False)

    print("\n" + "=" * 105)
    print("FULL BLIND2 FAMILY RESULTS — PRIMARY RMSE/MAE; R2 IS SECONDARY")
    print("=" * 105)
    print(full_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved to:", OUTPUT_DIR)
    print("\nBootstrap intervals quantify sampling uncertainty conditional on blind2;")
    print("they are not evidence that blind2 is an independent chemical population.")
    print("Blind2 is now consumed for this frozen family-level validation.")


if __name__ == "__main__":
    main()


