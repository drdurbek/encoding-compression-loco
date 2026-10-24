

"""Final leakage-free internal evaluation for the NADES Tg Final60 RF model.

Prerequisite
------------
Run TG_RF_DiversityGA_SubsetSize_Comparison.py first.  This script uses the
already selected Final60 descriptor set and the frozen group split.  The
55-row outer internal-evaluation set is accessed here for the first and only
time; it must not be used to change descriptors or RF hyperparameters.
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

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 0. PATHS
# ============================================================

BASE = data_path('TG')

TRAIN_PATH = data_path('TG/master_train_features.csv')
SPLIT_IN = data_path('TG/TG_clean_split_seed42.csv')
SELECTED_FEATURES_PATH = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('TG/TG_Final60_RF_Internal_Evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_OUT = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_fixed_group_split.csv')
PREDICTIONS_OUT = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_internal_evaluation_predictions.csv')
GROUP_PREDICTIONS_OUT = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_group_level_predictions.csv')
METRICS_OUT = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_internal_evaluation_metrics.csv')
PARITY_PNG = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_train_and_heldout_parity.png')
PARITY_PDF = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_train_and_heldout_parity.pdf')
EVALUATION_BUNDLE_OUT = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_outer_train_RF_bundle.joblib')
FINAL_BUNDLE_OUT = data_path('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_full_dataset_RF_bundle.joblib')


# ============================================================
# 1. FROZEN SETTINGS
# ============================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"       # Legacy column name; values are Tg in K.
GROUP_COLUMN = "Core_NADES_ID"   # Validation metadata, never a descriptor.

SEED = 42
N_BOOTSTRAP = 5000
CI_LEVEL = 0.95

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

# Identical RF hyperparameters to the previous Tg model.
RF_PARAMS = {
    "max_depth": 6,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 180,
    "random_state": 42,
    "n_jobs": -1,
}

plt.rcParams["font.family"] = "Arial"


# ============================================================
# 2. HELPERS
# ============================================================

def normalise_code(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def build_model():
    return RandomForestRegressor(**RF_PARAMS)


def regression_metrics(y_true, y_pred):
    residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_K": float(root_mean_squared_error(y_true, y_pred)),
        "mae_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(residual)),
        "mean_absolute_error_K": float(np.mean(np.abs(residual))),
    }


def group_bootstrap_ci(y_true, y_pred, groups, n_boot, seed, ci):
    """Bootstrap complete parent-NADES groups, never individual rows."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups, dtype=str)

    unique_groups = np.unique(groups)
    group_rows = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(seed)

    r2_values = []
    rmse_values = []
    mae_values = []

    for _ in range(n_boot):
        sampled_group_ids = rng.integers(0, len(unique_groups), size=len(unique_groups))
        row_indices = np.concatenate([group_rows[index] for index in sampled_group_ids])

        yt = y_true[row_indices]
        yp = y_pred[row_indices]
        if np.unique(yt).size < 2:
            continue

        r2_values.append(r2_score(yt, yp))
        rmse_values.append(root_mean_squared_error(yt, yp))
        mae_values.append(mean_absolute_error(yt, yp))

    if not r2_values:
        raise RuntimeError("Group bootstrap produced no valid resamples.")

    alpha = (1.0 - ci) / 2.0
    lo, hi = 100.0 * alpha, 100.0 * (1.0 - alpha)

    return {
        "n_bootstrap_used": len(r2_values),
        "r2_ci_low": float(np.percentile(r2_values, lo)),
        "r2_ci_high": float(np.percentile(r2_values, hi)),
        "rmse_ci_low_K": float(np.percentile(rmse_values, lo)),
        "rmse_ci_high_K": float(np.percentile(rmse_values, hi)),
        "mae_ci_low_K": float(np.percentile(mae_values, lo)),
        "mae_ci_high_K": float(np.percentile(mae_values, hi)),
    }


def save_parity_plot(y_train, pred_train, y_test, pred_test, out_png, out_pdf):
    all_values = np.concatenate([y_train, pred_train, y_test, pred_test])
    padding = 0.04 * (all_values.max() - all_values.min())
    lower = all_values.min() - padding
    upper = all_values.max() + padding

    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.scatter(
        y_train, pred_train,
        s=42, color="#298C8C", alpha=0.72,
        edgecolors="white", linewidths=0.35,
        label="Outer-train (n = 208)",
    )
    ax.scatter(
        y_test, pred_test,
        s=58, color="#E76F51", alpha=0.90,
        edgecolors="white", linewidths=0.45,
        label="Held-out parent NADES (n = 55)",
        zorder=3,
    )
    ax.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental Tg (K)", fontsize=14)
    ax.set_ylabel("Predicted Tg (K)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# 3. LOAD DATA AND RECREATE THE FROZEN OUTER SPLIT
# ============================================================

for path, description in [
    (TRAIN_PATH, "Tg master feature table"),
    (SPLIT_IN, "frozen grouped split"),
    (SELECTED_FEATURES_PATH, "Final60 selected descriptors"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")

data = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)
split_ref = pd.read_csv(SPLIT_IN, low_memory=False)
selected_table = pd.read_csv(SELECTED_FEATURES_PATH, low_memory=False)

required_data = {CODE_COLUMN, TARGET_COLUMN, GROUP_COLUMN}
if not required_data.issubset(data.columns):
    raise ValueError(f"Dataset must contain {sorted(required_data)}.")
if not {CODE_COLUMN, "split"}.issubset(split_ref.columns):
    raise ValueError("The split file must contain Code and split columns.")
if "feature" not in selected_table.columns:
    raise ValueError("The selected-descriptor file must contain a feature column.")

data[CODE_COLUMN] = normalise_code(data[CODE_COLUMN])
split_ref[CODE_COLUMN] = normalise_code(split_ref[CODE_COLUMN])
data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")

if data[TARGET_COLUMN].isna().any():
    raise ValueError("activity contains missing or non-numeric Tg values.")
if data[CODE_COLUMN].isna().any() or data[CODE_COLUMN].duplicated().any():
    raise ValueError("Code values must be unique and non-missing.")
if data[GROUP_COLUMN].isna().any():
    raise ValueError("Core_NADES_ID contains missing values.")

selected_features = (
    selected_table["feature"].dropna().astype(str).str.strip().drop_duplicates().tolist()
)
if len(selected_features) != 60:
    raise ValueError(
        "The Final60 descriptor file must contain exactly 10 unique features; "
        f"found {len(selected_features)}."
    )

missing_features = [feature for feature in selected_features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Selected features missing from master table: {missing_features[:20]}")

split_map = dict(zip(split_ref[CODE_COLUMN], split_ref["split"]))
split_label = data[CODE_COLUMN].map(split_map)
if split_label.isna().any():
    examples = data.loc[split_label.isna(), CODE_COLUMN].head(10).tolist()
    raise ValueError(f"Codes missing from split file: {examples}")

split_normalised = split_label.astype("string").str.strip().str.lower()
is_train = split_normalised.isin({"train", "outer_train", "training"}).to_numpy()
is_test = ~is_train

if is_train.sum() != 208 or is_test.sum() != 55:
    raise ValueError(
        "Unexpected outer split size. Expected 208 train and 55 held-out rows; "
        f"found {is_train.sum()} and {is_test.sum()}."
    )

X_raw = (
    data[selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
y = data[TARGET_COLUMN].to_numpy(dtype=float)
groups = data[GROUP_COLUMN].astype(str).to_numpy()
codes = data[CODE_COLUMN].astype(str).to_numpy()

X_train_raw = X_raw.loc[is_train]
X_test_raw = X_raw.loc[is_test]
y_train = y[is_train]
y_test = y[is_test]
groups_train = groups[is_train]
groups_test = groups[is_test]
codes_test = codes[is_test]

if set(groups_train).intersection(set(groups_test)):
    raise RuntimeError("Group leakage: parent NADES occurs in both outer partitions.")

pd.DataFrame({
    CODE_COLUMN: codes,
    TARGET_COLUMN: y,
    GROUP_COLUMN: groups,
    "split": np.where(is_train, "Outer_train", "Held_out"),
}).to_csv(SPLIT_OUT, index=False)


# ============================================================
# 4. FIT ON OUTER-TRAIN ONLY; PREDICT HELD-OUT GROUPS ONCE
# ============================================================

evaluation_imputer = SimpleImputer(strategy="median")
X_train = evaluation_imputer.fit_transform(X_train_raw)
X_test = evaluation_imputer.transform(X_test_raw)

evaluation_model = build_model()
evaluation_model.fit(X_train, y_train)

train_prediction = evaluation_model.predict(X_train)
test_prediction = evaluation_model.predict(X_test)

train_metrics = regression_metrics(y_train, train_prediction)
test_metrics = regression_metrics(y_test, test_prediction)
test_ci = group_bootstrap_ci(
    y_true=y_test,
    y_pred=test_prediction,
    groups=groups_test,
    n_boot=N_BOOTSTRAP,
    seed=SEED,
    ci=CI_LEVEL,
)


# ============================================================
# 5. SAVE PREDICTIONS, GROUP SUMMARY, METRICS, AND FIGURE
# ============================================================

prediction_table = pd.DataFrame({
    CODE_COLUMN: codes_test,
    GROUP_COLUMN: groups_test,
    "experimental_Tg_K": y_test,
    "predicted_Tg_K": test_prediction,
})
prediction_table["residual_predicted_minus_experimental_K"] = (
    prediction_table["predicted_Tg_K"] - prediction_table["experimental_Tg_K"]
)
prediction_table["absolute_error_K"] = np.abs(
    prediction_table["residual_predicted_minus_experimental_K"]
)
prediction_table.to_csv(PREDICTIONS_OUT, index=False)

group_table = (
    prediction_table
    .groupby(GROUP_COLUMN, as_index=False)
    .agg(
        n_concentrations=(CODE_COLUMN, "size"),
        mean_experimental_Tg_K=("experimental_Tg_K", "mean"),
        mean_predicted_Tg_K=("predicted_Tg_K", "mean"),
        mean_absolute_error_K=("absolute_error_K", "mean"),
    )
)
group_table.to_csv(GROUP_PREDICTIONS_OUT, index=False)

metrics_rows = []
for split_name, metrics in [("outer_train_apparent", train_metrics), ("held_out", test_metrics)]:
    for metric, value in metrics.items():
        metrics_rows.append({"split": split_name, "metric": metric, "value": value})
for metric, value in test_ci.items():
    metrics_rows.append({"split": "held_out_group_bootstrap_95CI", "metric": metric, "value": value})
metrics_rows.extend([
    {"split": "metadata", "metric": "n_outer_train", "value": int(len(y_train))},
    {"split": "metadata", "metric": "n_held_out", "value": int(len(y_test))},
    {"split": "metadata", "metric": "n_held_out_parent_NADES", "value": int(len(np.unique(groups_test)))},
    {"split": "metadata", "metric": "n_selected_descriptors", "value": int(len(selected_features))},
])
pd.DataFrame(metrics_rows).to_csv(METRICS_OUT, index=False)

save_parity_plot(
    y_train, train_prediction,
    y_test, test_prediction,
    PARITY_PNG, PARITY_PDF,
)


# ============================================================
# 6. SAVE THE EVALUATION MODEL AND FINAL DEPLOYMENT MODEL
# ============================================================

evaluation_bundle = {
    "model": evaluation_model,
    "imputer": evaluation_imputer,
    "selected_features": selected_features,
    "target": "Tg_K",
    "target_source_column": TARGET_COLUMN,
    "group_column": GROUP_COLUMN,
    "rf_parameters": RF_PARAMS,
    "model_purpose": "Outer-train model used once for frozen Final60 held-out evaluation.",
}
joblib.dump(evaluation_bundle, EVALUATION_BUNDLE_OUT)

# Only after the held-out result above is recorded, refit a deployment model
# using all 263 labelled rows. This model must not replace the evaluation model.
final_imputer = SimpleImputer(strategy="median")
X_full = final_imputer.fit_transform(X_raw)
final_model = build_model()
final_model.fit(X_full, y)

final_bundle = {
    "model": final_model,
    "imputer": final_imputer,
    "selected_features": selected_features,
    "target": "Tg_K",
    "target_source_column": TARGET_COLUMN,
    "group_column": GROUP_COLUMN,
    "rf_parameters": RF_PARAMS,
    "training_samples": int(len(y)),
    "model_purpose": "Deployment model retrained on all labelled data after held-out evaluation.",
}
joblib.dump(final_bundle, FINAL_BUNDLE_OUT)


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 94)
print("FINAL60 RANDOM-FOREST: GROUPED HELD-OUT Tg EVALUATION")
print("=" * 94)
print(f"Outer-train rows / parent NADES : {len(y_train)} / {len(np.unique(groups_train))}")
print(f"Held-out rows / parent NADES    : {len(y_test)} / {len(np.unique(groups_test))}")
print(f"Selected descriptors             : {len(selected_features)}")
print("\nHeld-out result (unseen parent NADES):")
print(
    f"R2 = {test_metrics['r2']:.4f} "
    f"[{test_ci['r2_ci_low']:.4f}, {test_ci['r2_ci_high']:.4f}]"
)
print(
    f"RMSE = {test_metrics['rmse_K']:.3f} K "
    f"[{test_ci['rmse_ci_low_K']:.3f}, {test_ci['rmse_ci_high_K']:.3f}] K"
)
print(
    f"MAE = {test_metrics['mae_K']:.3f} K "
    f"[{test_ci['mae_ci_low_K']:.3f}, {test_ci['mae_ci_high_K']:.3f}] K"
)
print(f"Mean error = {test_metrics['mean_error_K']:.3f} K")
print("\nSaved:")
for path in [
    SPLIT_OUT, PREDICTIONS_OUT, GROUP_PREDICTIONS_OUT, METRICS_OUT,
    PARITY_PNG, PARITY_PDF, EVALUATION_BUNDLE_OUT, FINAL_BUNDLE_OUT,
]:
    print(" -", path)
print("\nFinal60 was fixed before this held-out evaluation.")
print("No held-out result is used for model or descriptor selection.")


