
"""Train-only grouped OOF benchmark of regression model families for Tg.

All models use the same frozen, DAGA-selected Final60 descriptor set.  The
outer held-out 55 rows (seven parent NADES) are never accessed: this script is
only a model-family diagnostic on the 208-row outer-training partition.

Because Final60 was selected with a Random-Forest-guided GA, this is not a
fully nested contest to choose a new final learner.  It is a transparent
robustness comparison of fixed, modest model-family baselines.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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

OUTPUT_DIR = data_path('TG/TG_Final60_Model_Family_Benchmark_GroupOOF')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_OUT = data_path('TG/TG_Final60_Model_Family_Benchmark_GroupOOF/TG_Final60_model_family_group_OOF_metrics.csv')
FOLD_METRICS_OUT = data_path('TG/TG_Final60_Model_Family_Benchmark_GroupOOF/TG_Final60_model_family_group_OOF_fold_metrics.csv')
PREDICTIONS_OUT = data_path('TG/TG_Final60_Model_Family_Benchmark_GroupOOF/TG_Final60_model_family_group_OOF_predictions_long.csv')
SETTINGS_OUT = data_path('TG/TG_Final60_Model_Family_Benchmark_GroupOOF/TG_Final60_model_family_benchmark_settings.csv')
FIGURE_PNG = data_path('TG/TG_Final60_Model_Family_Benchmark_GroupOOF/TG_Final60_model_family_group_OOF_metrics.png')
FIGURE_PDF = data_path('TG/TG_Final60_Model_Family_Benchmark_GroupOOF/TG_Final60_model_family_group_OOF_metrics.pdf')


# ============================================================
# 1. FIXED SETTINGS
# ============================================================

SEED = 42
N_SPLITS = 5
CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"          # continuous Tg in K; legacy name
GROUP_COLUMN = "Core_NADES_ID"      # metadata only, never a descriptor

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

# Fixed, intentionally modest comparator settings; no hyperparameter search
# is performed in this script.
RIDGE_PARAMS = {"alpha": 1.0}

# Exact Final60 RF configuration, retained as the main-model reference.
RANDOM_FOREST_PARAMS = {
    "max_depth": 6,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 180,
    "n_jobs": -1,
}

EXTRA_TREES_PARAMS = {
    "max_depth": 6,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 300,
    "max_features": 1.0,
    "n_jobs": -1,
}

SVR_RBF_PARAMS = {
    "C": 10.0,
    "epsilon": 0.10,
    "gamma": "scale",
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": -1,
}

MODEL_ORDER = [
    "Ridge regression",
    "Random forest",
    "Extra trees",
    "SVR-RBF",
    "XGBoost",
]

MODEL_COLOURS = {
    "Ridge regression": "#4C78A8",
    "Random forest": "#298C8C",
    "Extra trees": "#54A24B",
    "SVR-RBF": "#E17C05",
    "XGBoost": "#8C6BB1",
}


# ============================================================
# 2. HELPERS
# ============================================================

def normalise_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def make_model(name: str, fold_number: int):
    """Return a fresh fold-specific leakage-free pipeline."""
    model_seed = SEED + fold_number

    if name == "Ridge regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(**RIDGE_PARAMS)),
        ])

    if name == "Random forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                **RANDOM_FOREST_PARAMS,
                random_state=model_seed,
            )),
        ])

    if name == "Extra trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                **EXTRA_TREES_PARAMS,
                random_state=model_seed,
            )),
        ])

    if name == "SVR-RBF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVR(**SVR_RBF_PARAMS)),
        ])

    if name == "XGBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(
                **XGB_PARAMS,
                random_state=model_seed,
            )),
        ])

    raise ValueError(f"Unknown model name: {name}")


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_K": float(root_mean_squared_error(y_true, y_pred)),
        "mae_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(residual)),
    }


def evaluate_model_oof(model_name, X, y, groups, codes, splits):
    oof_prediction = np.full(len(y), np.nan, dtype=float)
    fold_rows = []

    for fold_number, (fit_idx, val_idx) in enumerate(splits, start=1):
        model = make_model(model_name, fold_number)
        model.fit(X.iloc[fit_idx], y[fit_idx])
        prediction = model.predict(X.iloc[val_idx])
        oof_prediction[val_idx] = prediction

        fold_rows.append({
            "model": model_name,
            "fold": fold_number,
            "n_train_rows": int(len(fit_idx)),
            "n_validation_rows": int(len(val_idx)),
            "n_train_parent_NADES": int(len(np.unique(groups[fit_idx]))),
            "n_validation_parent_NADES": int(len(np.unique(groups[val_idx]))),
            **regression_metrics(y[val_idx], prediction),
        })

    if np.isnan(oof_prediction).any():
        raise RuntimeError(f"Incomplete OOF predictions for {model_name}.")

    combined = regression_metrics(y, oof_prediction)
    prediction_table = pd.DataFrame({
        CODE_COLUMN: codes,
        GROUP_COLUMN: groups,
        "experimental_Tg_K": y,
        "predicted_Tg_K": oof_prediction,
        "model": model_name,
    })
    prediction_table["residual_predicted_minus_experimental_K"] = (
        prediction_table["predicted_Tg_K"] - prediction_table["experimental_Tg_K"]
    )

    return combined, pd.DataFrame(fold_rows), prediction_table


def style_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=11)


# ============================================================
# 3. LOAD THE FROZEN FINAL60 OUTER-TRAIN PARTITION
# ============================================================

for path, description in [
    (TRAIN_PATH, "Tg master feature table"),
    (SPLIT_PATH, "frozen grouped split"),
    (SELECTED_FEATURES_PATH, "Final60 descriptor file"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")

data = pd.read_csv(TRAIN_PATH, na_values=NA_VALUES, low_memory=False)
split_ref = pd.read_csv(SPLIT_PATH, low_memory=False)
selected = pd.read_csv(SELECTED_FEATURES_PATH, low_memory=False)

required_data = {CODE_COLUMN, TARGET_COLUMN, GROUP_COLUMN}
if not required_data.issubset(data.columns):
    raise ValueError(f"Master table must contain {sorted(required_data)}.")
if not {CODE_COLUMN, "split"}.issubset(split_ref.columns):
    raise ValueError("Frozen split must contain Code and split columns.")
if "feature" not in selected.columns:
    raise ValueError("Final60 descriptor file must contain a feature column.")

data[CODE_COLUMN] = normalise_code(data[CODE_COLUMN])
split_ref[CODE_COLUMN] = normalise_code(split_ref[CODE_COLUMN])
data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
if data[TARGET_COLUMN].isna().any():
    raise ValueError("activity contains missing/non-numeric Tg values.")
if data[GROUP_COLUMN].isna().any():
    raise ValueError("Core_NADES_ID contains missing values.")
if data[CODE_COLUMN].isna().any() or data[CODE_COLUMN].duplicated().any():
    raise ValueError("Code values must be unique and non-missing.")

features = (
    selected["feature"].dropna().astype(str).str.strip().drop_duplicates().tolist()
)
if len(features) != 60:
    raise ValueError(f"Expected 60 Final60 descriptors, found {len(features)}.")
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Final60 features missing from master: {missing_features[:10]}")

split_map = dict(zip(split_ref[CODE_COLUMN], split_ref["split"]))
split_label = data[CODE_COLUMN].map(split_map)
if split_label.isna().any():
    examples = data.loc[split_label.isna(), CODE_COLUMN].head(10).tolist()
    raise ValueError(f"Codes missing from frozen split: {examples}")

is_train = split_label.astype("string").str.strip().str.lower().isin(
    {"train", "outer_train", "training"}
).to_numpy()
if is_train.sum() != 208 or (~is_train).sum() != 55:
    raise ValueError(
        "Expected frozen split of 208 outer-train and 55 held-out rows; "
        f"found {is_train.sum()} and {(~is_train).sum()}."
    )

# From here on, hold-out rows are never passed to a model or metric.
X = (
    data.loc[is_train, features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .reset_index(drop=True)
)
y = data.loc[is_train, TARGET_COLUMN].to_numpy(dtype=float)
groups = data.loc[is_train, GROUP_COLUMN].astype(str).to_numpy()
codes = data.loc[is_train, CODE_COLUMN].astype(str).to_numpy()

n_groups = len(np.unique(groups))
if n_groups < N_SPLITS:
    raise ValueError(f"Only {n_groups} parent NADES groups; cannot form {N_SPLITS} folds.")

splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))

settings = pd.DataFrame([
    {"model": "Ridge regression", "parameters": str(RIDGE_PARAMS)},
    {"model": "Random forest", "parameters": str(RANDOM_FOREST_PARAMS)},
    {"model": "Extra trees", "parameters": str(EXTRA_TREES_PARAMS)},
    {"model": "SVR-RBF", "parameters": str(SVR_RBF_PARAMS)},
    {"model": "XGBoost", "parameters": str(XGB_PARAMS)},
])
settings.to_csv(SETTINGS_OUT, index=False)

print("=" * 96)
print("TG FINAL60 TRAIN-ONLY GROUP-OOF MODEL-FAMILY BENCHMARK")
print("=" * 96)
print(f"Fixed descriptors               : {len(features)}")
print(f"Outer-train rows / parent NADES: {len(y)} / {n_groups}")
print(f"Validation                      : {N_SPLITS}-fold GroupKFold")
print("Outer held-out partition        : not loaded or used")


# ============================================================
# 4. EVALUATE FIXED MODEL FAMILIES
# ============================================================

summary_rows = []
fold_tables = []
prediction_tables = []

for model_name in MODEL_ORDER:
    print("\n" + "-" * 96)
    print(f"Model: {model_name}")
    combined, fold_table, prediction_table = evaluate_model_oof(
        model_name, X, y, groups, codes, splits
    )

    summary_row = {
        "model": model_name,
        "n_descriptors": len(features),
        "combined_oof_r2": combined["r2"],
        "combined_oof_rmse_K": combined["rmse_K"],
        "combined_oof_mae_K": combined["mae_K"],
        "combined_oof_mean_error_K": combined["mean_error_K"],
    }
    for metric in ["r2", "rmse_K", "mae_K", "mean_error_K"]:
        summary_row[f"mean_fold_{metric}"] = float(fold_table[metric].mean())
        summary_row[f"sd_fold_{metric}"] = float(fold_table[metric].std(ddof=1))

    summary_rows.append(summary_row)
    fold_tables.append(fold_table)
    prediction_tables.append(prediction_table)

    print(
        f"Combined OOF: R2={combined['r2']:.4f} | "
        f"RMSE={combined['rmse_K']:.3f} K | "
        f"MAE={combined['mae_K']:.3f} K | "
        f"bias={combined['mean_error_K']:.3f} K"
    )

summary = pd.DataFrame(summary_rows)
summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
summary = summary.sort_values("model", ignore_index=True)
summary.to_csv(METRICS_OUT, index=False)
pd.concat(fold_tables, ignore_index=True).to_csv(FOLD_METRICS_OUT, index=False)
pd.concat(prediction_tables, ignore_index=True).to_csv(PREDICTIONS_OUT, index=False)


# ============================================================
# 5. FIGURE: OOF MODEL-FAMILY COMPARISON
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.6))
plot_specs = [
    ("combined_oof_r2", "Grouped OOF $R^2$", True),
    ("combined_oof_rmse_K", "Grouped OOF RMSE (K)", False),
    ("combined_oof_mae_K", "Grouped OOF MAE (K)", False),
]

for ax, (column, x_label, higher_is_better) in zip(axes, plot_specs):
    table = summary.sort_values(column, ascending=not higher_is_better).reset_index(drop=True)
    positions = np.arange(len(table))
    colours = [MODEL_COLOURS[name] for name in table["model"].astype(str)]
    ax.barh(positions, table[column], color=colours)
    ax.set_yticks(positions)
    ax.set_yticklabels(table["model"].astype(str), fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=13)
    style_axis(ax)

fig.suptitle(
    "Fixed Final60 descriptor set: train-only grouped OOF model-family benchmark",
    fontsize=15, y=1.01,
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
print("TG FINAL60 TRAIN-ONLY GROUP-OOF MODEL-FAMILY SUMMARY")
print("=" * 96)
print(
    summary[[
        "model", "combined_oof_r2", "combined_oof_rmse_K", "combined_oof_mae_K",
        "combined_oof_mean_error_K", "mean_fold_r2", "sd_fold_r2",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}")
)

print("\nSaved:")
for path in [
    METRICS_OUT, FOLD_METRICS_OUT, PREDICTIONS_OUT, SETTINGS_OUT,
    FIGURE_PNG, FIGURE_PDF,
]:
    print(" -", path)

print("\nDone. The 55-row grouped held-out partition was not used.")


