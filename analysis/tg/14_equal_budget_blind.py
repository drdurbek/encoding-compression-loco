
"""
TG FAMILY READOUT — FROZEN BLINDED CONCENTRATION VALIDATION
===========================================================

This script tests the controlled, equal-budget descriptor-family readouts on
the Tg blind set.  It is a blinded concentration-series validation, NOT an
external chemical-generalization test: blind rows may belong to parent NADES
formulations represented in the labelled training data at other concentrations.

Frozen before blind Tg values are joined
-----------------------------------------
* descriptor families and budget K=8;
* train-only technical cleaning, supervised ranking, and correlation pruning;
* RF readout (300 trees, depth 6);
* all 263 labelled Tg rows for fitting.

Inputs, all inside .../07092026New/TG
---------------------------------------
    master_train_features.csv
    master_blind_features.csv
    blind_with_activity.csv     # Code, activity; joined only after prediction
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

import matplotlib.pyplot as plt
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
BLIND_FEATURE_PATH = data_path('TG/master_blind_features.csv')
BLIND_LABEL_PATH = data_path('TG/blind_with_activity.csv')
OUTPUT_DIR = data_path('TG/TG_Family_Blind_Concentration_Validation_RF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FROZEN CONTROLLED-READOUT SETTINGS
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
    "random_state": SEED,
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

plt.rcParams["font.family"] = "Arial"


# =============================================================================
# 3. HELPERS
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
    train: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Technical cleaning fit only on the labelled training table."""
    X = train.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    log = {"input_columns": int(X.shape[1])}

    keep = X.isna().mean(axis=0) <= MAX_TRAIN_MISSING_FRACTION
    log["removed_over_missing"] = int((~keep).sum())
    X = X.loc[:, keep]
    keep = X.nunique(axis=0, dropna=True) > 1
    log["removed_constant"] = int((~keep).sum())
    X = X.loc[:, keep]
    duplicated = X.T.duplicated(keep="first")
    log["removed_duplicate"] = int(duplicated.sum())
    X = X.loc[:, ~duplicated]
    log["retained_columns"] = int(X.shape[1])
    return X, log


def select_features_train_only(
    train: pd.DataFrame,
    y: np.ndarray,
    candidate_columns: list[str],
    seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
    """Train-only regression prefilter and K=8 correlation-pruned selection."""
    X_clean, log = clean_train_features(train, candidate_columns)
    if X_clean.shape[1] < DESCRIPTOR_BUDGET:
        raise ValueError(f"Only {X_clean.shape[1]} clean features remain; need K={DESCRIPTOR_BUDGET}.")

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index
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

    ranks = {feature: rank for rank, feature in enumerate(selected, start=1)}
    scores["selected"] = scores["feature"].isin(selected)
    scores["selection_rank"] = scores["feature"].map(ranks)
    return selected, scores, log


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_K": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(y_pred - y_true)),
    }


# =============================================================================
# 4. FROZEN FAMILY MODELS AND BLINDED CONCENTRATION TEST
# =============================================================================

def main() -> None:
    for path, description in [
        (TRAIN_PATH, "Labelled Tg master file"),
        (BLIND_FEATURE_PATH, "Blind Tg feature master file"),
        (BLIND_LABEL_PATH, "Blind Tg label file"),
    ]:
        require_file(path, description)

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    blind = pd.read_csv(BLIND_FEATURE_PATH, low_memory=False)
    if {CODE_COLUMN, TARGET_COLUMN} - set(train.columns):
        raise ValueError("Training Tg master must contain Code and activity.")
    if CODE_COLUMN not in blind.columns or blind[CODE_COLUMN].duplicated().any():
        raise ValueError("Blind Tg feature master must contain unique Code values.")

    y_train = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training Tg values must be numeric and non-missing.")
    y_train = y_train.to_numpy(float)

    family_map = build_family_map(train.columns)
    missing = set().union(*family_map.values()) - set(blind.columns)
    if missing:
        raise ValueError(f"Blind master lacks descriptors, e.g. {sorted(missing)[:10]}")

    print("=" * 100)
    print("TG FAMILY READOUT — FROZEN BLINDED CONCENTRATION VALIDATION")
    print("=" * 100)
    print(f"Labelled training rows  : {len(train)}")
    print(f"Blind concentration rows: {len(blind)}")
    print(f"Fixed descriptor budget : {DESCRIPTOR_BUDGET} per family")
    print("Blind Tg values are not accessed until all predictions are generated.\n")

    metric_rows, prediction_rows, selection_rows, cleaning_rows = [], [], [], []

    for family_number, family in enumerate(FAMILY_ORDER):
        selected, scores, log = select_features_train_only(
            train, y_train, family_map[family], SEED + family_number
        )
        X_train = train.loc[:, selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_blind = blind.loc[:, selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_blind = imputer.transform(X_blind)
        model = RandomForestRegressor(**{**RF_PARAMS, "random_state": SEED + family_number})
        model.fit(X_train, y_train)
        prediction = model.predict(X_blind)

        scores.insert(0, "family", family)
        scores.insert(1, "family_label", SHORT_LABELS[family])
        selection_rows.append(scores)
        cleaning_rows.append({"family": family, "family_label": SHORT_LABELS[family], "n_selected": len(selected), **log})
        prediction_rows.append(pd.DataFrame({
            CODE_COLUMN: blind[CODE_COLUMN].to_numpy(),
            "family": family,
            "family_label": SHORT_LABELS[family],
            "predicted_Tg_K": prediction,
        }))
        print(f"Frozen {SHORT_LABELS[family]:22s}: selected {len(selected)} train-only descriptors; blind Tg predicted.")

    predictions = pd.concat(prediction_rows, ignore_index=True)

    # Labels are read and joined only after the six family prediction vectors exist.
    blind_labels = pd.read_csv(BLIND_LABEL_PATH, low_memory=False)
    if not {CODE_COLUMN, TARGET_COLUMN}.issubset(blind_labels.columns):
        raise ValueError("blind_with_activity.csv must contain Code and activity.")
    blind_labels = blind_labels[[CODE_COLUMN, TARGET_COLUMN]].copy()
    blind_labels[TARGET_COLUMN] = pd.to_numeric(blind_labels[TARGET_COLUMN], errors="coerce")
    if blind_labels[CODE_COLUMN].duplicated().any() or blind_labels[TARGET_COLUMN].isna().any():
        raise ValueError("Blind Tg labels must have unique Code values and numeric activity.")
    if set(blind[CODE_COLUMN]) != set(blind_labels[CODE_COLUMN]):
        raise ValueError("Blind feature and blind-label Code sets do not match exactly.")

    predictions = predictions.merge(blind_labels, on=CODE_COLUMN, how="left", validate="many_to_one")
    predictions["error_K"] = predictions["predicted_Tg_K"] - predictions[TARGET_COLUMN]
    predictions["absolute_error_K"] = predictions["error_K"].abs()
    predictions = predictions.sort_values(["family", CODE_COLUMN]).reset_index(drop=True)

    for family in FAMILY_ORDER:
        frame = predictions.loc[predictions["family"] == family]
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
    metrics.to_csv(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_metrics.csv'), index=False)
    predictions.to_csv(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_predictions.csv'), index=False)
    pd.concat(selection_rows, ignore_index=True).to_csv(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_selected_features.csv'), index=False)
    pd.DataFrame(cleaning_rows).to_csv(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_cleaning_log.csv'), index=False)

    # Figure A: external concentration-series RMSE and MAE.
    ordered = metrics.set_index("family").loc[FAMILY_ORDER]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x, width = np.arange(len(FAMILY_ORDER)), 0.35
    for offset, column, label, color in [
        (-width / 2, "rmse_K", "RMSE", "#298c8c"),
        (width / 2, "mae_K", "MAE", "#f39b7f"),
    ]:
        bars = ax.bar(x + offset, ordered[column], width, label=label, color=color)
        for bar, value in zip(bars, ordered[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [SHORT_LABELS[f] for f in FAMILY_ORDER], rotation=28, ha="right")
    ax.set_ylabel("Blind concentration-series error (K)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_metrics.png'), dpi=600, bbox_inches="tight")
    fig.savefig(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_metrics.pdf'), bbox_inches="tight")
    plt.close(fig)

    # Figure B: parity plot of each family on the same blind concentrations.
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    y_all = predictions[TARGET_COLUMN].to_numpy(float)
    lo, hi = y_all.min() - 3, y_all.max() + 3
    for ax, family in zip(axes.ravel(), FAMILY_ORDER):
        frame = predictions.loc[predictions["family"] == family]
        ax.scatter(frame[TARGET_COLUMN], frame["predicted_Tg_K"], s=48, color="#298c8c", alpha=0.85)
        ax.plot([lo, hi], [lo, hi], "--", color="#777777", linewidth=1.2)
        result = ordered.loc[family]
        ax.set_title(SHORT_LABELS[family], fontsize=13)
        ax.text(0.04, 0.96, f"RMSE = {result['rmse_K']:.2f} K\n$R^2$ = {result['r2']:.3f}", transform=ax.transAxes, va="top", fontsize=10)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.supxlabel("Experimental Tg (K)", fontsize=15)
    fig.supylabel("Predicted Tg (K)", fontsize=15)
    fig.tight_layout()
    fig.savefig(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_parity.png'), dpi=600, bbox_inches="tight")
    fig.savefig(data_path('TG/TG_Family_Blind_Concentration_Validation_RF/blind_concentration_family_parity.pdf'), bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 100)
    print("BLINDED CONCENTRATION-SERIES FAMILY RESULTS")
    print("=" * 100)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved to:", OUTPUT_DIR)
    print("\nThis blind set is now consumed for this family-level concentration-series comparison.")
    print("Do not use the results to alter descriptor budget, family definitions, or RF parameters.")


if __name__ == "__main__":
    main()


