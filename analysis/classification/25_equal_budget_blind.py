
"""
CLASSIFICATION FAMILY READOUT — FROZEN EXTERNAL BLIND VALIDATION
=================================================================

Purpose
-------
Confirm, exactly once on the external blind mixtures, the controlled
family-readout result obtained by NADES_CrossTask_Family_Multisplit_RF.py.

This is NOT a new model-selection experiment.  The following are fixed before
blind labels are consulted:
    * descriptor families;
    * descriptor budget (K = 8 per family);
    * train-only supervised ranking and correlation pruning;
    * Random-Forest classifier and its hyperparameters;
    * operational threshold 0.50 for threshold-dependent metrics.

Data flow
---------
1. All feature selection, imputation, and model fitting use ONLY the 639-row
   labelled training master table.
2. The frozen family models predict the blind feature table.
3. Only after prediction are Code/activity labels from blind_with_activity.csv
   joined to compute external metrics.

Inputs (same project root)
--------------------------
    master_train_features.csv
    master_blind_features.csv
    blind_with_activity.csv       # Code, activity only; never used in fitting

Outputs
-------
Classification_Family_Blind_Validation_RF/
    blind_family_metrics.csv
    blind_family_predictions.csv
    blind_family_selected_features.csv
    blind_family_cleaning_log.csv
    blind_family_metrics.png/.pdf
    blind_family_probability_heatmap.png/.pdf
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. PATHS
# =============================================================================

BASE = data_path('')

TRAIN_PATH = data_path('master_train_features.csv')
BLIND_FEATURE_PATH = data_path('master_blind_features.csv')
BLIND_LABEL_PATH = data_path('blind_with_activity.csv')

OUTPUT_DIR = data_path('Classification_Family_Blind_Validation_RF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. FROZEN READOUT SETTINGS
# =============================================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"

DESCRIPTOR_BUDGET = 8
PREFILTER_TOP_N = 250
CORRELATION_THRESHOLD = 0.95
MAX_TRAIN_MISSING_FRACTION = 0.95
THRESHOLD = 0.50
SEED = 20260801

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
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
# 3. VALIDATION AND FAMILY HELPERS
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
    mapping = {family: [] for family in FAMILY_ORDER}
    for column in columns:
        family = family_of_column(column)
        if family in {
            "Component_A", "Component_B", "Weighted_mixture",
            "Arithmetic_mean", "AB_interactions",
        }:
            mapping[family].append(column)
    mapping["Component_A_plus_B"] = (
        mapping["Component_A"] + mapping["Component_B"]
    )
    return mapping


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def numeric_train_clean(
    train: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Technical filtering fit only on labelled training rows."""
    X = train.loc[:, candidate_columns].apply(pd.to_numeric, errors="coerce")
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
    y_train: np.ndarray,
    candidate_columns: list[str],
    seed: int,
) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
    """Deterministic supervised K=8 selection on labelled training data only."""
    X_clean, log = numeric_train_clean(train, candidate_columns)
    if X_clean.shape[1] < DESCRIPTOR_BUDGET:
        raise ValueError(
            f"Only {X_clean.shape[1]} clean descriptors remain; K={DESCRIPTOR_BUDGET}."
        )

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X_clean), index=X_clean.index, columns=X_clean.columns
    )

    y_train = np.asarray(y_train, dtype=int)
    mean0, mean1 = X.loc[y_train == 0].mean(), X.loc[y_train == 1].mean()
    var0 = X.loc[y_train == 0].var(ddof=1)
    var1 = X.loc[y_train == 1].var(ddof=1)
    pooled_sd = np.sqrt((var0 + var1) / 2.0).replace(0.0, np.nan)
    abs_d = ((mean1 - mean0).abs() / pooled_sd).fillna(0.0)

    screen_columns = abs_d.nlargest(min(PREFILTER_TOP_N, X.shape[1])).index.tolist()
    X_screen = X.loc[:, screen_columns]
    mi = mutual_info_classif(X_screen.to_numpy(), y_train, random_state=seed)
    auc_star = np.array([
        max(roc_auc_score(y_train, X_screen[column]), 1.0 - roc_auc_score(y_train, X_screen[column]))
        for column in screen_columns
    ])
    score = (
        0.50 * percentile_rank(abs_d.loc[screen_columns].to_numpy())
        + 0.30 * percentile_rank(mi)
        + 0.20 * percentile_rank(auc_star)
    )

    scores = pd.DataFrame({
        "feature": screen_columns,
        "abs_cohen_d": abs_d.loc[screen_columns].to_numpy(),
        "mutual_information": mi,
        "auc_star": auc_star,
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
        raise ValueError(
            f"Correlation pruning retained {len(selected)} descriptors, not K={DESCRIPTOR_BUDGET}."
        )

    rank_map = {feature: rank for rank, feature in enumerate(selected, start=1)}
    scores["selected"] = scores["feature"].isin(selected)
    scores["selection_rank"] = scores["feature"].map(rank_map)
    return selected, scores, log


def blind_metrics(y_true: np.ndarray, probability_nades: np.ndarray) -> dict[str, float | int]:
    y_pred = (probability_nades >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc_nades": float(roc_auc_score(y_true, probability_nades)),
        "ap_nades": float(average_precision_score(y_true, probability_nades)),
        "ap_non_nades": float(
            average_precision_score((y_true == 0).astype(int), 1.0 - probability_nades)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc_at_0_50": float(matthews_corrcoef(y_true, y_pred)),
        "precision_nades": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_nades": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_non_nades": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# =============================================================================
# 4. FIT ON LABELLED TRAINING DATA; PREDICT BLIND FEATURES
# =============================================================================

def main() -> None:
    for path, description in [
        (TRAIN_PATH, "Labelled training master file"),
        (BLIND_FEATURE_PATH, "Blind feature master file"),
        (BLIND_LABEL_PATH, "Blind label file"),
    ]:
        require_file(path, description)

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    blind = pd.read_csv(BLIND_FEATURE_PATH, low_memory=False)

    if {CODE_COLUMN, TARGET_COLUMN} - set(train.columns):
        raise ValueError("Training master file must contain Code and activity.")
    y_train = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    if y_train.isna().any() or set(y_train.astype(int).unique()) != {0, 1}:
        raise ValueError("Training activity must be binary 0/1 without missing values.")
    y_train = y_train.astype(int).to_numpy()

    if CODE_COLUMN not in blind.columns or blind[CODE_COLUMN].duplicated().any():
        raise ValueError("Blind feature master must contain unique Code values.")

    # This check uses feature names only; blind activity is not loaded here.
    family_map = build_family_map(train.columns)
    missing_feature_columns = set().union(*family_map.values()) - set(blind.columns)
    if missing_feature_columns:
        raise ValueError(
            "Blind feature master lacks training descriptors, e.g.: "
            f"{sorted(missing_feature_columns)[:10]}"
        )

    print("=" * 100)
    print("CLASSIFICATION FAMILY READOUT — FROZEN EXTERNAL BLIND VALIDATION")
    print("=" * 100)
    print(f"Labelled training mixtures : {len(train)}")
    print(f"Blind feature mixtures     : {len(blind)}")
    print(f"Fixed descriptor budget    : {DESCRIPTOR_BUDGET} per family")
    print(f"Fixed RF threshold         : {THRESHOLD:.2f}")
    print("Blind labels are not accessed until all blind probabilities are produced.\n")

    metric_rows, prediction_rows, selection_rows, cleaning_rows = [], [], [], []
    probability_matrix: dict[str, np.ndarray] = {}

    for family_number, family in enumerate(FAMILY_ORDER):
        candidate_columns = family_map[family]
        selected, scores, log = select_features_train_only(
            train, y_train, candidate_columns, SEED + family_number
        )

        X_train = train.loc[:, selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_blind = blind.loc[:, selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_blind = imputer.transform(X_blind)

        model = RandomForestClassifier(
            **{**RF_PARAMS, "random_state": SEED + family_number}
        )
        model.fit(X_train, y_train)
        probability_nades = model.predict_proba(X_blind)[:, 1]
        probability_matrix[family] = probability_nades

        scores.insert(0, "family", family)
        scores.insert(1, "family_label", SHORT_LABELS[family])
        selection_rows.append(scores)
        cleaning_rows.append({
            "family": family, "family_label": SHORT_LABELS[family],
            "n_selected": len(selected), **log,
        })
        prediction_rows.append(pd.DataFrame({
            CODE_COLUMN: blind[CODE_COLUMN].to_numpy(),
            "family": family,
            "family_label": SHORT_LABELS[family],
            "probability_nades": probability_nades,
            "probability_non_nades": 1.0 - probability_nades,
            "predicted_activity_at_0_50": (probability_nades >= THRESHOLD).astype(int),
            "predicted_class_at_0_50": np.where(probability_nades >= THRESHOLD, "NADES", "non-NADES"),
        }))
        print(f"Frozen {SHORT_LABELS[family]:22s}: selected {len(selected)} train-only descriptors; blind probabilities generated.")

    predictions = pd.concat(prediction_rows, ignore_index=True)

    # -------------------------------------------------------------------------
    # Labels enter here, after all blind predictions have been constructed.
    # -------------------------------------------------------------------------
    blind_labels = pd.read_csv(BLIND_LABEL_PATH, low_memory=False)
    required_labels = {CODE_COLUMN, TARGET_COLUMN}
    if not required_labels.issubset(blind_labels.columns):
        raise ValueError("blind_with_activity.csv must contain Code and activity.")
    blind_labels = blind_labels[[CODE_COLUMN, TARGET_COLUMN]].copy()
    blind_labels[TARGET_COLUMN] = pd.to_numeric(blind_labels[TARGET_COLUMN], errors="coerce")
    if blind_labels[CODE_COLUMN].duplicated().any() or blind_labels[TARGET_COLUMN].isna().any():
        raise ValueError("Blind labels must have unique Code values and non-missing activity.")
    if set(blind_labels[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
        raise ValueError("Blind activity must contain both classes encoded as 0/1.")
    blind_labels[TARGET_COLUMN] = blind_labels[TARGET_COLUMN].astype(int)

    if set(blind[CODE_COLUMN]) != set(blind_labels[CODE_COLUMN]):
        raise ValueError("Blind feature and label Code sets do not match exactly.")

    predictions = predictions.merge(blind_labels, on=CODE_COLUMN, how="left", validate="many_to_one")
    predictions = predictions.sort_values(["family", CODE_COLUMN]).reset_index(drop=True)

    for family in FAMILY_ORDER:
        frame = predictions.loc[predictions["family"] == family]
        metrics = blind_metrics(
            frame[TARGET_COLUMN].to_numpy(dtype=int),
            frame["probability_nades"].to_numpy(dtype=float),
        )
        metric_rows.append({
            "family": family,
            "family_label": SHORT_LABELS[family],
            "n_blind": len(frame),
            "threshold": THRESHOLD,
            **metrics,
        })

    metrics = pd.DataFrame(metric_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    cleaning = pd.DataFrame(cleaning_rows)

    metrics.to_csv(data_path('Classification_Family_Blind_Validation_RF/blind_family_metrics.csv'), index=False)
    predictions.to_csv(data_path('Classification_Family_Blind_Validation_RF/blind_family_predictions.csv'), index=False)
    selections.to_csv(data_path('Classification_Family_Blind_Validation_RF/blind_family_selected_features.csv'), index=False)
    cleaning.to_csv(data_path('Classification_Family_Blind_Validation_RF/blind_family_cleaning_log.csv'), index=False)

    # =============================================================================
    # 5. FIGURE A — FAMILY-LEVEL EXTERNAL METRICS
    # =============================================================================
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(FAMILY_ORDER))
    width = 0.24
    ordered = metrics.set_index("family").loc[FAMILY_ORDER]
    for offset, column, label, color in [
        (-width, "roc_auc_nades", "ROC–AUC", "#298c8c"),
        (0, "ap_non_nades", "AP (non-NADES)", "#f39b7f"),
        (width, "mcc_at_0_50", "MCC at $\\tau$ = 0.50", "#4c78a8"),
    ]:
        bars = ax.bar(x + offset, ordered[column], width, label=label, color=color)
        for bar, value in zip(bars, ordered[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [SHORT_LABELS[f] for f in FAMILY_ORDER], rotation=28, ha="right")
    ax.set_ylabel("External blind-set performance", fontsize=15)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_path('Classification_Family_Blind_Validation_RF/blind_family_metrics.png'), dpi=600, bbox_inches="tight")
    fig.savefig(data_path('Classification_Family_Blind_Validation_RF/blind_family_metrics.pdf'), bbox_inches="tight")
    plt.close(fig)

    # =============================================================================
    # 6. FIGURE B — FAMILY PROBABILITIES FOR EACH BLIND MIXTURE
    # =============================================================================
    codes = blind[CODE_COLUMN].to_numpy()
    heat = np.vstack([probability_matrix[family] for family in FAMILY_ORDER])
    fig, ax = plt.subplots(figsize=(16, 5.2))
    image = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(FAMILY_ORDER)), [SHORT_LABELS[f] for f in FAMILY_ORDER])
    ax.set_xticks(np.arange(len(codes)), codes, rotation=0)
    ax.set_xlabel("Blind-set sample code", fontsize=14)
    ax.set_ylabel("Descriptor family", fontsize=14)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Predicted probability of NADES", fontsize=13)
    ax.set_title("Frozen family-model predictions on the blind set", fontsize=15)
    fig.tight_layout()
    fig.savefig(data_path('Classification_Family_Blind_Validation_RF/blind_family_probability_heatmap.png'), dpi=600, bbox_inches="tight")
    fig.savefig(data_path('Classification_Family_Blind_Validation_RF/blind_family_probability_heatmap.pdf'), bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 100)
    print("EXTERNAL BLIND-SET FAMILY RESULTS")
    print("=" * 100)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved to:", OUTPUT_DIR)
    print("\nInterpretation constraint: this blind set is now consumed for this family comparison.")
    print("Do not use its metrics to change the family ranking, descriptor budget, or RF settings.")


if __name__ == "__main__":
    main()


