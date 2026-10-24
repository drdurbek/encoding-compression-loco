
"""
TG pure-NADES LOGO: paired bootstrap contrasts of Weighted mixture vs families.

This is a post-processing script.  It DOES NOT refit feature selection,
imputation, Random Forest, or Ridge models, and it DOES NOT read blind2.

It reads the already frozen leave-one-parent-out (LOGO) OOF predictions from
the pure-NADES control and compares the Weighted_mixture representation with
each other descriptor family using the *same* bootstrap resamples of parent
NADES groups.  Therefore every contrast is paired.

Positive values always favour Weighted_mixture:

    weighted_advantage_rmse_K = RMSE(other family) - RMSE(Weighted mixture)
    weighted_advantage_mae_K  = MAE(other family)  - MAE(Weighted mixture)
    weighted_advantage_mse_skill = skill(Weighted mixture) - skill(other)

Here MSE skill is calculated against the fold-specific train-mean baseline
stored in the original OOF table.  This preserves the exact LOGO baseline used
when each parent NADES was predicted.

Required input created by TG_PureNADES_Control_LOCO_Blind2.py:
    TG/blind2/TG_PureNADES_Control_LOCO_Blind2/
        pure_nades_logo_oof_predictions.csv

Output:
    pure_nades_logo_paired_family_bootstrap_contrasts.csv
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

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS AND FROZEN SETTINGS
# =============================================================================

DEFAULT_TG_BASE = (
    r"D:\NDSU\NADES\Non_NADES_project"
    r"\new9022025\07092026New\TG"
)

# Optional command-line-independent override, useful outside the original
# Windows project directory, e.g. TG_BASE=/path/to/TG python script.py.
BASE = data_path('TG')
CONTROL_DIR = data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2')

OOF_PREDICTIONS_PATH = (
    data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_logo_oof_predictions.csv')
)
CONTRASTS_OUT = (
    data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2/pure_nades_logo_paired_family_bootstrap_contrasts.csv')
)

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"
GROUP_COLUMN = "parent_group"
MODEL_COLUMN = "model"
FAMILY_COLUMN = "family"
PREDICTION_COLUMN = "predicted_Tg_K"
BASELINE_COLUMN = "baseline_train_mean_K"

REFERENCE_FAMILY = "Weighted_mixture"
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "5000"))
SEED = 20260803

FAMILY_ORDER = [
    "Component_A",
    "Component_B",
    "Component_A_plus_B",
    "Weighted_mixture",
    "Arithmetic_mean",
    "AB_interactions",
]

FAMILY_LABELS = {
    "Component_A": "Component A",
    "Component_B": "Component B",
    "Component_A_plus_B": "A + B components",
    "Weighted_mixture": "Weighted mixture",
    "Arithmetic_mean": "Arithmetic mean",
    "AB_interactions": "AB interactions",
}


# =============================================================================
# 2. VALIDATION AND METRICS
# =============================================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def validate_oof_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    """Validate one frozen OOF prediction for every code/model/family cell."""
    required = {
        CODE_COLUMN,
        TARGET_COLUMN,
        GROUP_COLUMN,
        MODEL_COLUMN,
        FAMILY_COLUMN,
        PREDICTION_COLUMN,
        BASELINE_COLUMN,
    }
    require_columns(oof, required, OOF_PREDICTIONS_PATH.name)

    checked = oof.copy()
    checked[CODE_COLUMN] = checked[CODE_COLUMN].astype("string").str.strip()
    checked[GROUP_COLUMN] = checked[GROUP_COLUMN].astype("string").str.strip()
    checked[MODEL_COLUMN] = checked[MODEL_COLUMN].astype("string").str.strip()
    checked[FAMILY_COLUMN] = checked[FAMILY_COLUMN].astype("string").str.strip()

    text_columns = [CODE_COLUMN, GROUP_COLUMN, MODEL_COLUMN, FAMILY_COLUMN]
    for column in text_columns:
        if checked[column].isna().any() or checked[column].eq("").any():
            raise ValueError(f"OOF table contains missing values in {column!r}.")

    numeric_columns = [TARGET_COLUMN, PREDICTION_COLUMN, BASELINE_COLUMN]
    for column in numeric_columns:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
        if checked[column].isna().any() or not np.isfinite(checked[column]).all():
            raise ValueError(f"OOF table contains missing or non-finite values in {column!r}.")

    duplicated = checked.duplicated(
        subset=[CODE_COLUMN, MODEL_COLUMN, FAMILY_COLUMN], keep=False
    )
    if duplicated.any():
        examples = checked.loc[
            duplicated,
            [CODE_COLUMN, MODEL_COLUMN, FAMILY_COLUMN],
        ].head(10)
        raise ValueError(
            "OOF table must contain exactly one prediction per Code/model/family.\n"
            f"Examples of duplicated cells:\n{examples.to_string(index=False)}"
        )

    models = set(checked[MODEL_COLUMN])
    if not models:
        raise ValueError("OOF table contains no model rows.")
    families = set(checked[FAMILY_COLUMN])
    if REFERENCE_FAMILY not in families:
        raise ValueError(
            f"The reference family {REFERENCE_FAMILY!r} is absent from the OOF table."
        )

    return checked


def mse_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_prediction: np.ndarray,
) -> float:
    """Skill relative to each LOGO fold's train-mean baseline."""
    model_mse = float(np.mean((y_true - y_pred) ** 2))
    baseline_mse = float(np.mean((y_true - baseline_prediction) ** 2))
    if baseline_mse <= np.finfo(float).eps:
        return np.nan
    return 1.0 - (model_mse / baseline_mse)


def family_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_prediction: np.ndarray,
) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "mse_skill_vs_fold_train_mean": mse_skill(
            y_true,
            y_pred,
            baseline_prediction,
        ),
    }


def paired_draw_advantages(
    y_true: np.ndarray,
    weighted_prediction: np.ndarray,
    other_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Paired cluster bootstrap; each resample is identical for both families."""
    unique_groups = np.unique(groups)
    positions_by_group = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)

    draws = {
        "weighted_advantage_rmse_K": [],
        "weighted_advantage_mae_K": [],
        "weighted_advantage_mse_skill": [],
    }

    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled_positions = np.concatenate(
            [positions_by_group[group] for group in sampled_groups]
        )

        sample_y = y_true[sampled_positions]
        sample_weighted = weighted_prediction[sampled_positions]
        sample_other = other_prediction[sampled_positions]
        sample_baseline = baseline_prediction[sampled_positions]

        weighted = family_metrics(sample_y, sample_weighted, sample_baseline)
        other = family_metrics(sample_y, sample_other, sample_baseline)

        # Positive quantities always mean that Weighted mixture is better.
        draws["weighted_advantage_rmse_K"].append(
            other["rmse_K"] - weighted["rmse_K"]
        )
        draws["weighted_advantage_mae_K"].append(
            other["mae_K"] - weighted["mae_K"]
        )
        draws["weighted_advantage_mse_skill"].append(
            weighted["mse_skill_vs_fold_train_mean"]
            - other["mse_skill_vs_fold_train_mean"]
        )

    return {
        metric: np.asarray(values, dtype=float)
        for metric, values in draws.items()
    }


def directional_conclusion(ci_low: float, ci_high: float) -> str:
    if np.isfinite(ci_low) and ci_low > 0.0:
        return "Weighted mixture better"
    if np.isfinite(ci_high) and ci_high < 0.0:
        return "Comparator family better"
    return "Inconclusive at 95% bootstrap interval"


# =============================================================================
# 3. PAIRED CONTRASTS
# =============================================================================

def one_model_prediction_matrix(
    oof: pd.DataFrame,
    model: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a code-aligned family-prediction matrix for one readout model."""
    subset = oof.loc[oof[MODEL_COLUMN] == model].copy()
    available_families = set(subset[FAMILY_COLUMN])
    ordered_families = [
        family for family in FAMILY_ORDER
        if family in available_families
    ]
    unexpected_families = sorted(available_families - set(ordered_families))
    ordered_families.extend(unexpected_families)

    expected_cells = subset[CODE_COLUMN].nunique() * len(ordered_families)
    if len(subset) != expected_cells:
        counts = subset.groupby(CODE_COLUMN)[FAMILY_COLUMN].nunique()
        bad_codes = counts.loc[counts != len(ordered_families)].head(10).index.tolist()
        raise ValueError(
            f"{model}: not every code has one prediction for every family. "
            f"Examples: {bad_codes}"
        )

    metadata = subset[
        [CODE_COLUMN, TARGET_COLUMN, GROUP_COLUMN, BASELINE_COLUMN]
    ].drop_duplicates()
    if len(metadata) != subset[CODE_COLUMN].nunique():
        raise ValueError(
            f"{model}: target, group, or fold-specific baseline differs between "
            "families for one or more codes."
        )

    predictions = subset.pivot(
        index=CODE_COLUMN,
        columns=FAMILY_COLUMN,
        values=PREDICTION_COLUMN,
    )
    matrix = metadata.set_index(CODE_COLUMN).join(predictions, how="inner")
    matrix = matrix.sort_index()

    if matrix[ordered_families].isna().any().any():
        raise ValueError(f"{model}: missing family predictions after alignment.")
    if REFERENCE_FAMILY not in matrix.columns:
        raise ValueError(f"{model}: missing {REFERENCE_FAMILY!r} predictions.")

    return matrix, ordered_families


def compute_paired_contrasts(oof: pd.DataFrame) -> pd.DataFrame:
    """Return paired Weighted-vs-family bootstrap contrasts for every model."""
    rows: list[dict[str, object]] = []

    for model_number, model in enumerate(sorted(oof[MODEL_COLUMN].unique())):
        matrix, families = one_model_prediction_matrix(oof, model)
        y_true = matrix[TARGET_COLUMN].to_numpy(float)
        baseline = matrix[BASELINE_COLUMN].to_numpy(float)
        groups = matrix[GROUP_COLUMN].astype(str).to_numpy()
        n_groups = int(pd.Series(groups).nunique())
        bootstrap_unit = (
            "system (= unique parent group)"
            if n_groups == len(matrix)
            else "parent_group"
        )

        weighted_prediction = matrix[REFERENCE_FAMILY].to_numpy(float)
        for family_number, other_family in enumerate(families):
            if other_family == REFERENCE_FAMILY:
                continue

            other_prediction = matrix[other_family].to_numpy(float)
            point_weighted = family_metrics(y_true, weighted_prediction, baseline)
            point_other = family_metrics(y_true, other_prediction, baseline)
            point_estimates = {
                "weighted_advantage_rmse_K": (
                    point_other["rmse_K"] - point_weighted["rmse_K"]
                ),
                "weighted_advantage_mae_K": (
                    point_other["mae_K"] - point_weighted["mae_K"]
                ),
                "weighted_advantage_mse_skill": (
                    point_weighted["mse_skill_vs_fold_train_mean"]
                    - point_other["mse_skill_vs_fold_train_mean"]
                ),
            }
            draws = paired_draw_advantages(
                y_true=y_true,
                weighted_prediction=weighted_prediction,
                other_prediction=other_prediction,
                baseline_prediction=baseline,
                groups=groups,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED + 1000 * model_number + 10 * family_number,
            )

            for metric, values in draws.items():
                valid = values[np.isfinite(values)]
                ci_low = float(np.percentile(valid, 2.5)) if len(valid) else np.nan
                ci_high = float(np.percentile(valid, 97.5)) if len(valid) else np.nan
                point = float(point_estimates[metric])
                rows.append({
                    "model": model,
                    "reference_family": REFERENCE_FAMILY,
                    "reference_family_label": FAMILY_LABELS.get(
                        REFERENCE_FAMILY,
                        REFERENCE_FAMILY,
                    ),
                    "comparator_family": other_family,
                    "comparator_family_label": FAMILY_LABELS.get(
                        other_family,
                        other_family,
                    ),
                    "comparison": (
                        f"{FAMILY_LABELS.get(REFERENCE_FAMILY, REFERENCE_FAMILY)} "
                        f"vs {FAMILY_LABELS.get(other_family, other_family)}"
                    ),
                    "metric": metric,
                    "point_estimate": point,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "positive_favors": "Weighted mixture",
                    "bootstrap_fraction_favoring_weighted": (
                        float(np.mean(valid > 0.0)) if len(valid) else np.nan
                    ),
                    "conclusion": directional_conclusion(ci_low, ci_high),
                    "n_systems": int(len(matrix)),
                    "n_parent_groups": n_groups,
                    "bootstrap_unit": bootstrap_unit,
                    "n_bootstrap_requested": N_BOOTSTRAP,
                    "n_bootstrap_valid": int(len(valid)),
                })

    return pd.DataFrame(rows)


# =============================================================================
# 4. MAIN
# =============================================================================

def main() -> None:
    if N_BOOTSTRAP < 100:
        raise ValueError("N_BOOTSTRAP must be at least 100.")

    require_file(OOF_PREDICTIONS_PATH, "Frozen pure-NADES LOGO OOF predictions")
    oof = validate_oof_predictions(pd.read_csv(OOF_PREDICTIONS_PATH, low_memory=False))
    contrasts = compute_paired_contrasts(oof)

    data_path('TG/blind2/TG_PureNADES_Control_LOCO_Blind2').mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(CONTRASTS_OUT, index=False)

    print("=" * 105)
    print("TG PURE-NADES LOGO — PAIRED BOOTSTRAP: WEIGHTED MIXTURE VS OTHER FAMILIES")
    print("=" * 105)
    print(f"Input OOF rows             : {len(oof)}")
    print(f"Readout models             : {', '.join(sorted(oof[MODEL_COLUMN].unique()))}")
    print(f"Reference family           : {FAMILY_LABELS[REFERENCE_FAMILY]}")
    print(f"Bootstrap resamples        : {N_BOOTSTRAP}")
    print("Positive contrast = better Weighted mixture representation.\n")

    primary = contrasts.loc[
        contrasts["metric"] == "weighted_advantage_rmse_K"
    ].copy()
    print("Primary RMSE contrasts:")
    print(
        primary[
            [
                "model",
                "comparator_family_label",
                "point_estimate",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_fraction_favoring_weighted",
                "conclusion",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved:\n - {CONTRASTS_OUT}")
    print(
        "\nThis is post-processing of frozen LOGO OOF predictions only; "
        "no model, descriptor set, or blind2 result was changed."
    )


if __name__ == "__main__":
    main()


