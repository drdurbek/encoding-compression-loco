
"""LOCO chemical-extrapolation validation for the frozen Tg Final60 RF model.

For each chemical component, every Tg mixture containing that component is
held out.  A Random Forest is then fitted only to mixtures containing none of
that component and predicts the held-out mixtures.  This tests extrapolation
to a chemically unseen ingredient.

Final60 descriptors and RF hyperparameters were frozen before this script.
No feature selection, GA optimisation, or outer-held-out model choice occurs
here.  LOCO is a complementary chemical-extrapolation analysis; it is not the
primary grouped held-out evaluation.
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


warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = "Arial"


# ============================================================
# 0. PATHS
# ============================================================

BASE = data_path('TG')

MASTER_PATH = data_path('TG/master_train_features.csv')
ORIGINAL_PATH = data_path('TG/original.csv')
SELECTED_FEATURES_PATH = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('TG/TG_Final60_RF_LOCO')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = data_path('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_component_results.csv')
PREDICTIONS_OUT = data_path('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_sample_predictions.csv')
SUMMARY_OUT = data_path('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_summary.csv')
R2_FIGURE_PNG = data_path('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_R2_per_component.png')
R2_FIGURE_PDF = data_path('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_R2_per_component.pdf')
RMSE_FIGURE_PNG = data_path('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_RMSE_per_component.png')
RMSE_FIGURE_PDF = data_path('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_RMSE_per_component.pdf')


# ============================================================
# 1. FROZEN SETTINGS
# ============================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"          # continuous Tg, K
GROUP_COLUMN = "Core_NADES_ID"      # reporting metadata only
COMPONENT_COLUMNS = ["A_id", "B_id", "C_id", "D_id"]
EXPECTED_DESCRIPTOR_COUNT = 60
MIN_TEST_SAMPLES = 5
SEED = 42

# Exact Final60 RF configuration.
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


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_K": float(root_mean_squared_error(y_true, y_pred)),
        "MAE_K": float(mean_absolute_error(y_true, y_pred)),
        "mean_error_K": float(np.mean(residual)),
    }


def save_figure(fig, png_path: Path, pdf_path: Path):
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def style_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=11)


# ============================================================
# 3. LOAD FROZEN FEATURES AND COMPONENT IDENTITIES
# ============================================================

for path, description in [
    (MASTER_PATH, "Tg master feature table"),
    (ORIGINAL_PATH, "original component-identity table"),
    (SELECTED_FEATURES_PATH, "Final60 selected-descriptor table"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")

master = pd.read_csv(MASTER_PATH, na_values=NA_VALUES, low_memory=False)
original = pd.read_csv(ORIGINAL_PATH, na_values=NA_VALUES, low_memory=False)
selected = pd.read_csv(SELECTED_FEATURES_PATH, low_memory=False)

required_master = {CODE_COLUMN, TARGET_COLUMN}
required_original = {CODE_COLUMN, *COMPONENT_COLUMNS}
if not required_master.issubset(master.columns):
    raise ValueError(f"Master table must contain {sorted(required_master)}.")
if not required_original.issubset(original.columns):
    raise ValueError(f"Original table must contain {sorted(required_original)}.")
if "feature" not in selected.columns:
    raise ValueError("Selected-descriptor table must contain a feature column.")

master[CODE_COLUMN] = normalise_code(master[CODE_COLUMN])
original[CODE_COLUMN] = normalise_code(original[CODE_COLUMN])

if master[CODE_COLUMN].duplicated().any():
    raise ValueError("Duplicate Codes in master_train_features.csv.")
if original[CODE_COLUMN].duplicated().any():
    raise ValueError("Duplicate Codes in original.csv.")

data = master.merge(
    original[[CODE_COLUMN, *COMPONENT_COLUMNS]],
    on=CODE_COLUMN,
    how="inner",
    validate="one_to_one",
)
if len(data) != len(master):
    missing_codes = sorted(set(master[CODE_COLUMN]) - set(data[CODE_COLUMN]))[:15]
    raise ValueError(
        "Some master rows have no component identity in original.csv. "
        f"Examples: {missing_codes}"
    )

data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
if data[TARGET_COLUMN].isna().any():
    raise ValueError("activity contains missing/non-numeric Tg values.")

selected_features = (
    selected["feature"].dropna().astype(str).str.strip().drop_duplicates().tolist()
)
if len(selected_features) != EXPECTED_DESCRIPTOR_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_DESCRIPTOR_COUNT} selected descriptors; "
        f"found {len(selected_features)}."
    )
missing_features = [feature for feature in selected_features if feature not in data.columns]
if missing_features:
    raise ValueError(f"Final60 descriptors missing from master table: {missing_features[:15]}")

X_raw = (
    data[selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .reset_index(drop=True)
)
y = data[TARGET_COLUMN].to_numpy(dtype=float)
codes = data[CODE_COLUMN].astype(str).to_numpy()

component_frame = data[COMPONENT_COLUMNS].copy()
for column in COMPONENT_COLUMNS:
    component_frame[column] = component_frame[column].astype("string").str.strip()

components = sorted(
    str(component)
    for component in pd.unique(pd.concat(
        [component_frame[column] for column in COMPONENT_COLUMNS],
        ignore_index=True,
    ).dropna())
    if str(component)
)

if GROUP_COLUMN in data.columns:
    core_nades = data[GROUP_COLUMN].astype(str).to_numpy()
else:
    core_nades = np.full(len(data), "not_available", dtype=object)

print("=" * 96)
print("TG FINAL60 RANDOM-FOREST LEAVE-ONE-COMPONENT-OUT (LOCO) VALIDATION")
print("=" * 96)
print(f"Labelled Tg rows                : {len(data)}")
print(f"Fixed Final60 descriptors       : {len(selected_features)}")
print(f"Unique chemical components      : {len(components)}")
print(f"Minimum held-out rows/component : {MIN_TEST_SAMPLES}")
print("Selection rule                  : Final60 and RF parameters were frozen before LOCO")


# ============================================================
# 4. LOCO: EACH COMPONENT IS COMPLETELY UNSEEN DURING FITTING
# ============================================================

component_records = []
prediction_tables = []

for component_number, component in enumerate(components, start=1):
    component_mask = np.zeros(len(data), dtype=bool)
    for column in COMPONENT_COLUMNS:
        component_mask |= component_frame[column].eq(component).fillna(False).to_numpy()

    test_idx = np.flatnonzero(component_mask)
    train_idx = np.flatnonzero(~component_mask)

    if len(test_idx) < MIN_TEST_SAMPLES:
        continue

    # All descriptor transformations are fitted on mixtures without the
    # component.  The held-out component is never present in model fitting.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_raw.iloc[train_idx])
    X_test = imputer.transform(X_raw.iloc[test_idx])

    model = build_model()
    model.fit(X_train, y[train_idx])
    prediction = model.predict(X_test)
    component_metrics = regression_metrics(y[test_idx], prediction)

    n_test_parent_nades = (
        len(np.unique(core_nades[test_idx]))
        if GROUP_COLUMN in data.columns else np.nan
    )
    n_train_parent_nades = (
        len(np.unique(core_nades[train_idx]))
        if GROUP_COLUMN in data.columns else np.nan
    )

    component_records.append({
        "component": component,
        "n_test": int(len(test_idx)),
        "n_train": int(len(train_idx)),
        "n_test_parent_NADES": n_test_parent_nades,
        "n_train_parent_NADES": n_train_parent_nades,
        "Tg_test_min_K": float(np.min(y[test_idx])),
        "Tg_test_max_K": float(np.max(y[test_idx])),
        "Tg_test_sd_K": float(np.std(y[test_idx], ddof=1)) if len(test_idx) > 1 else np.nan,
        **component_metrics,
    })

    per_sample = pd.DataFrame({
        "component_held_out": component,
        CODE_COLUMN: codes[test_idx],
        "experimental_Tg_K": y[test_idx],
        "predicted_Tg_K": prediction,
    })
    if GROUP_COLUMN in data.columns:
        per_sample[GROUP_COLUMN] = core_nades[test_idx]
    per_sample["residual_predicted_minus_experimental_K"] = (
        per_sample["predicted_Tg_K"] - per_sample["experimental_Tg_K"]
    )
    per_sample["absolute_error_K"] = np.abs(
        per_sample["residual_predicted_minus_experimental_K"]
    )
    prediction_tables.append(per_sample)

    print(
        f"{component_number:3d}/{len(components)} | {component:30s} | "
        f"n_test={len(test_idx):3d} | R2={component_metrics['R2']:.3f} | "
        f"RMSE={component_metrics['RMSE_K']:.2f} K | MAE={component_metrics['MAE_K']:.2f} K"
    )

if not component_records:
    raise RuntimeError("No component met MIN_TEST_SAMPLES.")

results = pd.DataFrame(component_records).sort_values("component", ignore_index=True)
results.to_csv(RESULTS_OUT, index=False)
pd.concat(prediction_tables, ignore_index=True).to_csv(PREDICTIONS_OUT, index=False)

summary = pd.DataFrame([{
    "n_labelled_rows": int(len(data)),
    "n_fixed_descriptors": int(len(selected_features)),
    "n_unique_components_total": int(len(components)),
    "n_components_evaluated": int(len(results)),
    "n_components_with_negative_R2": int((results["R2"] < 0).sum()),
    "mean_component_R2": float(results["R2"].mean()),
    "median_component_R2": float(results["R2"].median()),
    "mean_component_RMSE_K": float(results["RMSE_K"].mean()),
    "median_component_RMSE_K": float(results["RMSE_K"].median()),
    "mean_component_MAE_K": float(results["MAE_K"].mean()),
    "median_component_MAE_K": float(results["MAE_K"].median()),
    "mean_component_bias_K": float(results["mean_error_K"].mean()),
    "note": (
        "Per-component LOCO test sets overlap when a mixture contains multiple components; "
        "therefore component metrics are descriptive and must not be pooled as one global OOF score."
    ),
}])
summary.to_csv(SUMMARY_OUT, index=False)


# ============================================================
# 5. FIGURES: COMPONENT-WISE CHEMICAL EXTRAPOLATION
# ============================================================

plot_results = results.sort_values("R2", ascending=True, ignore_index=True)
fig_height = max(7.0, 0.34 * len(plot_results))
fig, ax = plt.subplots(figsize=(9.2, fig_height))
positions = np.arange(len(plot_results))

ax.barh(positions, plot_results["R2"], color="#298C8C")
ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.1, label="$R^2$ = 0")
ax.axvline(
    plot_results["R2"].mean(), color="#E76F51", linestyle="--", linewidth=1.2,
    label=f"Mean = {plot_results['R2'].mean():.3f}",
)
ax.set_yticks(positions)
ax.set_yticklabels(plot_results["component"], fontsize=10)
ax.set_xlabel("LOCO $R^2$ for held-out component", fontsize=14)
ax.set_ylabel("Chemically unseen component", fontsize=14)
ax.legend(frameon=False, fontsize=11, loc="lower right")
style_axis(ax)
fig.tight_layout()
save_figure(fig, R2_FIGURE_PNG, R2_FIGURE_PDF)
plt.show()
plt.close(fig)

plot_results = results.sort_values("RMSE_K", ascending=True, ignore_index=True)
fig_height = max(7.0, 0.34 * len(plot_results))
fig, ax = plt.subplots(figsize=(9.2, fig_height))
positions = np.arange(len(plot_results))

ax.barh(positions, plot_results["RMSE_K"], color="#298C8C")
ax.axvline(
    plot_results["RMSE_K"].mean(), color="#E76F51", linestyle="--", linewidth=1.2,
    label=f"Mean = {plot_results['RMSE_K'].mean():.2f} K",
)
ax.set_yticks(positions)
ax.set_yticklabels(plot_results["component"], fontsize=10)
ax.set_xlabel("LOCO RMSE for held-out component (K)", fontsize=14)
ax.set_ylabel("Chemically unseen component", fontsize=14)
ax.legend(frameon=False, fontsize=11, loc="lower right")
style_axis(ax)
fig.tight_layout()
save_figure(fig, RMSE_FIGURE_PNG, RMSE_FIGURE_PDF)
plt.show()
plt.close(fig)


# ============================================================
# 6. FINAL REPORT
# ============================================================

print("\n" + "=" * 96)
print("TG FINAL60 LOCO SUMMARY")
print("=" * 96)
print(f"Evaluated components       : {len(results)}")
print(f"Mean component R2          : {results['R2'].mean():.4f}")
print(f"Median component R2        : {results['R2'].median():.4f}")
print(f"Mean component RMSE        : {results['RMSE_K'].mean():.3f} K")
print(f"Median component RMSE      : {results['RMSE_K'].median():.3f} K")
print(f"Mean component MAE         : {results['MAE_K'].mean():.3f} K")
print(f"Components with R2 < 0     : {(results['R2'] < 0).sum()}/{len(results)}")

print("\nSaved:")
for path in [
    RESULTS_OUT, PREDICTIONS_OUT, SUMMARY_OUT,
    R2_FIGURE_PNG, R2_FIGURE_PDF, RMSE_FIGURE_PNG, RMSE_FIGURE_PDF,
]:
    print(" -", path)

print("\nDone. Final60 descriptors and RF settings remained frozen throughout LOCO.")

