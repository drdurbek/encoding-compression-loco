
"""Train-only group-OOF family probe for the fixed Tg Final60 representation.

Each descriptor family in the frozen 60-descriptor set is assessed as an
independent RF model.  The script also evaluates the complete Final60 set as
a reference.  It never fits on, predicts, loads, or reports the 55-row outer
held-out partition.

This is an exploratory mechanistic decomposition, not a new model-selection
step.  Final60, the grouped split, and the RF settings are fixed before the
analysis begins.
"""

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold


warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = "Arial"


# ============================================================
# 0. PATHS
# ============================================================

BASE = data_path('TG')

TRAIN_PATH = data_path('TG/master_train_features.csv')
SPLIT_PATH = data_path('TG/TG_clean_split_seed42.csv')
SELECTED_FEATURES_PATH = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('TG/TG_Final60_Family_Only_TrainOnly_OOF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_COUNTS_OUT = data_path('TG/TG_Final60_Family_Only_TrainOnly_OOF/TG_Final60_family_descriptor_counts.csv')
FOLD_METRICS_OUT = data_path('TG/TG_Final60_Family_Only_TrainOnly_OOF/TG_Final60_family_only_fold_metrics.csv')
SUMMARY_OUT = data_path('TG/TG_Final60_Family_Only_TrainOnly_OOF/TG_Final60_family_only_group_OOF_metrics.csv')
PREDICTIONS_OUT = data_path('TG/TG_Final60_Family_Only_TrainOnly_OOF/TG_Final60_family_only_group_OOF_predictions.csv')
FIGURE_PNG = data_path('TG/TG_Final60_Family_Only_TrainOnly_OOF/TG_Final60_family_only_group_OOF_metrics.png')
FIGURE_PDF = data_path('TG/TG_Final60_Family_Only_TrainOnly_OOF/TG_Final60_family_only_group_OOF_metrics.pdf')


# ============================================================
# 1. FROZEN SETTINGS
# ============================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"          # Tg in K; legacy target name
GROUP_COLUMN = "Core_NADES_ID"      # validation metadata only
N_SPLITS = 5
SEED = 42

# Identical learner to TG_Final60_RF_Internal_Evaluation.py.
RF_PARAMS = {
    "max_depth": 6,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 180,
    "random_state": SEED,
    "n_jobs": -1,
}

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

REFERENCE_LABEL = "Full Final60"


# ============================================================
# 2. HELPERS
# ============================================================

def normalise_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(**RF_PARAMS)


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_K": float(root_mean_squared_error(y_true, y_pred)),
        "mae_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(residual)),
    }


def oof_family_model(X_raw, y, groups, codes, feature_names, model_label, splits):
    """Fit imputer and RF separately in every group-held-out OOF fold."""
    X_subset = X_raw.loc[:, feature_names]
    oof_prediction = np.full(len(y), np.nan, dtype=float)
    fold_records = []

    for fold_number, (fit_idx, val_idx) in enumerate(splits, start=1):
        imputer = SimpleImputer(strategy="median")
        X_fit = imputer.fit_transform(X_subset.iloc[fit_idx])
        X_val = imputer.transform(X_subset.iloc[val_idx])

        model = build_model()
        model.fit(X_fit, y[fit_idx])
        prediction = model.predict(X_val)
        oof_prediction[val_idx] = prediction

        fold_metric = metrics(y[val_idx], prediction)
        fold_records.append({
            "model": model_label,
            "fold": fold_number,
            "n_features": len(feature_names),
            "n_train_rows": len(fit_idx),
            "n_validation_rows": len(val_idx),
            "n_train_parent_NADES": len(np.unique(groups[fit_idx])),
            "n_validation_parent_NADES": len(np.unique(groups[val_idx])),
            **fold_metric,
        })

    if np.isnan(oof_prediction).any():
        raise RuntimeError(f"OOF predictions are incomplete for {model_label}.")

    prediction_table = pd.DataFrame({
        CODE_COLUMN: codes,
        GROUP_COLUMN: groups,
        "experimental_Tg_K": y,
        "predicted_Tg_K": oof_prediction,
        "model": model_label,
        "n_features": len(feature_names),
    })
    prediction_table["residual_predicted_minus_experimental_K"] = (
        prediction_table["predicted_Tg_K"] - prediction_table["experimental_Tg_K"]
    )

    return metrics(y, oof_prediction), fold_records, prediction_table


def summary_row(label, n_features, combined, fold_records):
    fold_table = pd.DataFrame(fold_records)
    row = {
        "model": label,
        "n_descriptors": int(n_features),
        "combined_oof_r2": combined["r2"],
        "combined_oof_rmse_K": combined["rmse_K"],
        "combined_oof_mae_K": combined["mae_K"],
        "combined_oof_mean_error_K": combined["mean_error_K"],
    }
    for metric_name in ["r2", "rmse_K", "mae_K", "mean_error_K"]:
        row[f"mean_fold_{metric_name}"] = float(fold_table[metric_name].mean())
        row[f"sd_fold_{metric_name}"] = float(fold_table[metric_name].std(ddof=1))
    return row


def style_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=11)


# ============================================================
# 3. LOAD FIXED FINAL60 TRAIN PARTITION
# ============================================================

for path, description in [
    (TRAIN_PATH, "Tg master feature table"),
    (SPLIT_PATH, "frozen grouped split"),
    (SELECTED_FEATURES_PATH, "Final60 selected-descriptor table"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")

data = pd.read_csv(TRAIN_PATH, na_values=NA_VALUES, low_memory=False)
split_ref = pd.read_csv(SPLIT_PATH, low_memory=False)
selected = pd.read_csv(SELECTED_FEATURES_PATH, low_memory=False)

required_data = {CODE_COLUMN, TARGET_COLUMN, GROUP_COLUMN}
required_selected = {"feature", "family"}
if not required_data.issubset(data.columns):
    raise ValueError(f"Master table must contain {sorted(required_data)}.")
if not {CODE_COLUMN, "split"}.issubset(split_ref.columns):
    raise ValueError("Frozen split must contain Code and split columns.")
if not required_selected.issubset(selected.columns):
    raise ValueError(f"Selected descriptor table must contain {sorted(required_selected)}.")

data[CODE_COLUMN] = normalise_code(data[CODE_COLUMN])
split_ref[CODE_COLUMN] = normalise_code(split_ref[CODE_COLUMN])
data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")

if data[TARGET_COLUMN].isna().any():
    raise ValueError("activity contains missing/non-numeric Tg values.")
if data[GROUP_COLUMN].isna().any():
    raise ValueError("Core_NADES_ID contains missing values.")
if data[CODE_COLUMN].duplicated().any():
    raise ValueError("Code values must be unique.")

selected = selected[["feature", "family"]].dropna().copy()
selected["feature"] = selected["feature"].astype(str).str.strip()
selected["family"] = selected["family"].astype(str).str.strip()
selected = selected.drop_duplicates("feature", keep="first").reset_index(drop=True)

if len(selected) != 60:
    raise ValueError(f"Expected exactly 60 Final60 descriptors, found {len(selected)}.")

missing_features = [feature for feature in selected["feature"] if feature not in data.columns]
if missing_features:
    raise ValueError(f"Final60 features missing from master: {missing_features[:15]}")

split_map = dict(zip(split_ref[CODE_COLUMN], split_ref["split"]))
split_label = data[CODE_COLUMN].map(split_map)
if split_label.isna().any():
    raise ValueError("Some master Codes are absent from TG_clean_split_seed42.csv.")

is_train = split_label.astype("string").str.strip().str.lower().isin(
    {"train", "outer_train", "training"}
).to_numpy()

if is_train.sum() != 208:
    raise ValueError(f"Expected 208 outer-train rows, found {is_train.sum()}.")

# From this point onward the script uses outer-train rows only.
features = selected["feature"].tolist()
X_outer_train = (
    data.loc[is_train, features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .reset_index(drop=True)
)
y_outer_train = data.loc[is_train, TARGET_COLUMN].to_numpy(dtype=float)
groups_outer_train = data.loc[is_train, GROUP_COLUMN].astype(str).to_numpy()
codes_outer_train = data.loc[is_train, CODE_COLUMN].astype(str).to_numpy()

n_groups = len(np.unique(groups_outer_train))
if n_groups < N_SPLITS:
    raise ValueError(f"Only {n_groups} parent NADES groups; cannot use {N_SPLITS} folds.")

splits = list(GroupKFold(n_splits=N_SPLITS).split(
    X_outer_train, y_outer_train, groups_outer_train
))

family_counts = (
    selected.groupby("family", as_index=False)
    .agg(n_descriptors=("feature", "size"))
    .sort_values("family", ignore_index=True)
)
family_counts.to_csv(FAMILY_COUNTS_OUT, index=False)

print("=" * 96)
print("TG FINAL60 FAMILY-ONLY TRAIN-ONLY GROUP-OOF EXPERIMENT")
print("=" * 96)
print(f"Outer-train rows / parent NADES : {len(y_outer_train)} / {n_groups}")
print(f"Validation                      : {N_SPLITS}-fold GroupKFold")
print("Outer held-out partition        : not loaded into the experiment")
print("\nFixed Final60 family composition:")
print(family_counts.to_string(index=False))


# ============================================================
# 4. COMPLETE FINAL60 AND INDIVIDUAL-FAMILY OOF MODELS
# ============================================================

all_summary_rows = []
all_fold_records = []
all_prediction_tables = []

model_definitions = [(REFERENCE_LABEL, features)]
for family, family_table in selected.groupby("family", sort=True):
    model_definitions.append((family, family_table["feature"].tolist()))

for model_label, model_features in model_definitions:
    print("\n" + "-" * 96)
    print(f"Model: {model_label} | descriptors = {len(model_features)}")

    combined, fold_records, prediction_table = oof_family_model(
        X_raw=X_outer_train,
        y=y_outer_train,
        groups=groups_outer_train,
        codes=codes_outer_train,
        feature_names=model_features,
        model_label=model_label,
        splits=splits,
    )

    all_summary_rows.append(summary_row(
        model_label, len(model_features), combined, fold_records
    ))
    all_fold_records.extend(fold_records)
    all_prediction_tables.append(prediction_table)

    print(
        f"Combined OOF: R2={combined['r2']:.4f} | "
        f"RMSE={combined['rmse_K']:.3f} K | "
        f"MAE={combined['mae_K']:.3f} K | "
        f"bias={combined['mean_error_K']:.3f} K"
    )

summary = pd.DataFrame(all_summary_rows)
summary["is_full_Final60_reference"] = summary["model"].eq(REFERENCE_LABEL)
summary = summary.sort_values(
    ["is_full_Final60_reference", "combined_oof_rmse_K"],
    ascending=[False, True],
    ignore_index=True,
)
summary.to_csv(SUMMARY_OUT, index=False)
pd.DataFrame(all_fold_records).to_csv(FOLD_METRICS_OUT, index=False)
pd.concat(all_prediction_tables, ignore_index=True).to_csv(PREDICTIONS_OUT, index=False)


# ============================================================
# 5. FIGURE: FAMILY-ONLY OOF METRICS
# ============================================================

plot_table = summary.copy()
plot_table["display_label"] = plot_table.apply(
    lambda row: (
        f"{row['model']} (n={int(row['n_descriptors'])})"
        if row["model"] != REFERENCE_LABEL else "Full Final60 (n=60)"
    ),
    axis=1,
)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8))
panel_specs = [
    ("combined_oof_r2", "OOF $R^2$", True),
    ("combined_oof_rmse_K", "OOF RMSE (K)", False),
    ("combined_oof_mae_K", "OOF MAE (K)", False),
]

for ax, (column, label, higher_is_better) in zip(axes, panel_specs):
    ordering = plot_table.sort_values(column, ascending=not higher_is_better).reset_index(drop=True)
    colours = np.where(
        ordering["model"].eq(REFERENCE_LABEL), "#E76F51", "#298C8C"
    )
    positions = np.arange(len(ordering))
    ax.barh(positions, ordering[column], color=colours)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordering["display_label"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(label, fontsize=13)
    style_axis(ax)

    reference_value = float(
        ordering.loc[ordering["model"].eq(REFERENCE_LABEL), column].iloc[0]
    )
    ax.axvline(reference_value, color="#E76F51", linestyle="--", linewidth=1.1)

fig.suptitle(
    "Train-only grouped OOF performance of individual Final60 descriptor families",
    fontsize=15, y=1.01,
)
fig.text(
    0.5, -0.01,
    "Teal: individual family only; orange: full 60-descriptor reference",
    ha="center", fontsize=11,
)
fig.tight_layout()
fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight")
fig.savefig(FIGURE_PDF, bbox_inches="tight")
plt.show()
plt.close(fig)


# ============================================================
# 6. FINAL REPORT
# ============================================================

print("\n" + "=" * 96)
print("FINAL60 FAMILY-ONLY TRAIN-ONLY GROUP-OOF SUMMARY")
print("=" * 96)
print(
    summary[[
        "model", "n_descriptors", "combined_oof_r2", "combined_oof_rmse_K",
        "combined_oof_mae_K", "combined_oof_mean_error_K",
        "mean_fold_r2", "sd_fold_r2",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
)

print("\nSaved:")
for path in [
    FAMILY_COUNTS_OUT, FOLD_METRICS_OUT, SUMMARY_OUT, PREDICTIONS_OUT,
    FIGURE_PNG, FIGURE_PDF,
]:
    print(" -", path)

print("\nDone. The 55-row grouped held-out partition was not used.")


