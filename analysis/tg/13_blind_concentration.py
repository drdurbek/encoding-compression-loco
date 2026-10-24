

"""External blind-set evaluation for the frozen Tg Final60 RF workflow.

The RF settings and 60 descriptors are fixed before this script runs.  The
model is fitted on all labelled Tg data and predicts ``master_blind_features``.
The separate ``blind_with_activity.csv`` file supplies experimental Tg values
only after prediction, by joining on Code; labels never enter model fitting,
feature selection, model selection, or tuning.
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
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.family"] = "Arial"


# ============================================================
# 0. PATHS
# ============================================================

BASE = data_path('TG')

TRAIN_PATH = data_path('TG/master_train_features.csv')
BLIND_FEATURES_PATH = data_path('TG/master_blind_features.csv')
BLIND_LABELS_PATH = data_path('TG/blind_with_activity.csv')
SELECTED_FEATURES_PATH = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('TG/TG_Final60_RF_External_Blind_Evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_OUT = data_path('TG/TG_Final60_RF_External_Blind_Evaluation/TG_Final60_external_blind_predictions.csv')
METRICS_OUT = data_path('TG/TG_Final60_RF_External_Blind_Evaluation/TG_Final60_external_blind_metrics.csv')
PARITY_PNG = data_path('TG/TG_Final60_RF_External_Blind_Evaluation/TG_Final60_external_blind_parity.png')
PARITY_PDF = data_path('TG/TG_Final60_RF_External_Blind_Evaluation/TG_Final60_external_blind_parity.pdf')
MODEL_BUNDLE_OUT = data_path('TG/TG_Final60_RF_External_Blind_Evaluation/TG_Final60_full_train_RF_for_external_blind.joblib')


# ============================================================
# 1. FROZEN SETTINGS
# ============================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"          # continuous Tg in K
EXPECTED_DESCRIPTOR_COUNT = 60

# Exact Final60 RF configuration; do not tune on blind results.
RF_PARAMS = {
    "max_depth": 6,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 180,
    "random_state": 42,
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


def save_parity_plot(y_train, pred_train, y_blind, pred_blind, out_png, out_pdf):
    all_values = np.concatenate([y_train, pred_train, y_blind, pred_blind])
    padding = 0.04 * (all_values.max() - all_values.min())
    lower = all_values.min() - padding
    upper = all_values.max() + padding

    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.scatter(
        y_train, pred_train,
        s=38, color="#298C8C", alpha=0.58,
        edgecolors="white", linewidths=0.30,
        label=f"Labelled training data (n = {len(y_train)})",
    )
    ax.scatter(
        y_blind, pred_blind,
        s=64, color="#E76F51", alpha=0.92,
        edgecolors="white", linewidths=0.45,
        label=f"External blind set (n = {len(y_blind)})",
        zorder=3,
    )
    ax.plot([lower, upper], [lower, upper], "--", color="#777777", linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Experimental Tg (K)", fontsize=14)
    ax.set_ylabel("Predicted Tg (K)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# 3. LOAD TRAIN, BLIND, AND FROZEN DESCRIPTOR SET
# ============================================================

for path, description in [
    (TRAIN_PATH, "labelled Tg training table"),
    (BLIND_FEATURES_PATH, "external blind feature table"),
    (BLIND_LABELS_PATH, "external blind Tg label table"),
    (SELECTED_FEATURES_PATH, "Final60 selected-descriptor table"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")

train = pd.read_csv(TRAIN_PATH, na_values=NA_VALUES, low_memory=False)
blind = pd.read_csv(BLIND_FEATURES_PATH, na_values=NA_VALUES, low_memory=False)
selected = pd.read_csv(SELECTED_FEATURES_PATH, low_memory=False)

required_train = {CODE_COLUMN, TARGET_COLUMN}
required_blind = {CODE_COLUMN}
if not required_train.issubset(train.columns):
    raise ValueError(f"Training table must contain {sorted(required_train)}.")
if not required_blind.issubset(blind.columns):
    raise ValueError(
        "master_blind_features.csv must contain Code."
    )
if "feature" not in selected.columns:
    raise ValueError("Final60 selected-descriptor table must contain a feature column.")

train[CODE_COLUMN] = normalise_code(train[CODE_COLUMN])
blind[CODE_COLUMN] = normalise_code(blind[CODE_COLUMN])

for frame, name in [(train, "training")]:
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"{name} activity contains missing/non-numeric Tg values.")
    if frame[CODE_COLUMN].isna().any() or frame[CODE_COLUMN].duplicated().any():
        raise ValueError(f"{name} Code values must be unique and non-missing.")

if blind[CODE_COLUMN].isna().any() or blind[CODE_COLUMN].duplicated().any():
    raise ValueError("Blind feature-table Code values must be unique and non-missing.")

code_overlap = sorted(set(train[CODE_COLUMN]) & set(blind[CODE_COLUMN]))
if code_overlap:
    raise ValueError(
        "Training and blind tables overlap in Code values. "
        f"Examples: {code_overlap[:10]}"
    )

selected_features = (
    selected["feature"].dropna().astype(str).str.strip().drop_duplicates().tolist()
)
if len(selected_features) != EXPECTED_DESCRIPTOR_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_DESCRIPTOR_COUNT} Final60 descriptors, "
        f"found {len(selected_features)}."
    )

missing_train = [feature for feature in selected_features if feature not in train.columns]
missing_blind = [feature for feature in selected_features if feature not in blind.columns]
if missing_train:
    raise ValueError(f"Features missing from labelled training table: {missing_train[:15]}")
if missing_blind:
    raise ValueError(f"Features missing from blind table: {missing_blind[:15]}")

X_train_raw = (
    train[selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
X_blind_raw = (
    blind[selected_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)
y_train = train[TARGET_COLUMN].to_numpy(dtype=float)


# ============================================================
# 4. FIT ON ALL LABELLED Tg DATA; PREDICT EXTERNAL BLIND ONCE
# ============================================================

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_blind = imputer.transform(X_blind_raw)

model = RandomForestRegressor(**RF_PARAMS)
model.fit(X_train, y_train)

train_prediction = model.predict(X_train)
blind_prediction = model.predict(X_blind)

train_metrics = regression_metrics(y_train, train_prediction)

# Labels are intentionally loaded only after blind probabilities/predictions
# already exist. They are used solely for post hoc external comparison.
blind_labels = pd.read_csv(BLIND_LABELS_PATH, na_values=NA_VALUES, low_memory=False)
required_labels = {CODE_COLUMN, TARGET_COLUMN}
if not required_labels.issubset(blind_labels.columns):
    raise ValueError(
        "blind_with_activity.csv must contain Code and activity "
        "(experimental Tg in K)."
    )
blind_labels = blind_labels[[CODE_COLUMN, TARGET_COLUMN]].copy()
blind_labels[CODE_COLUMN] = normalise_code(blind_labels[CODE_COLUMN])
blind_labels[TARGET_COLUMN] = pd.to_numeric(blind_labels[TARGET_COLUMN], errors="coerce")
if blind_labels[CODE_COLUMN].isna().any() or blind_labels[CODE_COLUMN].duplicated().any():
    raise ValueError("blind_with_activity.csv Code values must be unique and non-missing.")
if blind_labels[TARGET_COLUMN].isna().any():
    raise ValueError("blind_with_activity.csv activity contains missing/non-numeric Tg values.")

blind_evaluation = blind[[CODE_COLUMN]].merge(
    blind_labels,
    on=CODE_COLUMN,
    how="left",
    validate="one_to_one",
)
if blind_evaluation[TARGET_COLUMN].isna().any():
    missing = blind_evaluation.loc[
        blind_evaluation[TARGET_COLUMN].isna(), CODE_COLUMN
    ].head(10).tolist()
    raise ValueError(
        "Some blind feature rows have no experimental Tg in blind_with_activity.csv. "
        f"Examples: {missing}"
    )

y_blind = blind_evaluation[TARGET_COLUMN].to_numpy(dtype=float)
blind_metrics = regression_metrics(y_blind, blind_prediction)


# ============================================================
# 5. SAVE EXTERNAL PREDICTIONS, METRICS, MODEL, AND FIGURE
# ============================================================

output = pd.DataFrame({
    CODE_COLUMN: blind[CODE_COLUMN].astype(str),
    "experimental_Tg_K": y_blind,
    "predicted_Tg_K": blind_prediction,
})
output["residual_predicted_minus_experimental_K"] = (
    output["predicted_Tg_K"] - output["experimental_Tg_K"]
)
output["absolute_error_K"] = np.abs(output["residual_predicted_minus_experimental_K"])
output["rank_by_absolute_error"] = output["absolute_error_K"].rank(
    method="first", ascending=False
).astype(int)

numeric_code = pd.to_numeric(output[CODE_COLUMN], errors="coerce")
output = (
    output.assign(_numeric_code=numeric_code)
    .sort_values(["_numeric_code", CODE_COLUMN], na_position="last")
    .drop(columns="_numeric_code")
    .reset_index(drop=True)
)
output.to_csv(PREDICTIONS_OUT, index=False)

metrics_table = pd.DataFrame([
    {"split": "training_resubstitution_diagnostic", "n_samples": len(y_train), **train_metrics},
    {"split": "external_blind", "n_samples": len(y_blind), **blind_metrics},
])
metrics_table.to_csv(METRICS_OUT, index=False)

bundle = {
    "model": model,
    "imputer": imputer,
    "selected_features": selected_features,
    "rf_parameters": RF_PARAMS,
    "model_purpose": "Final60 RF fitted on all labelled Tg rows before external blind evaluation.",
    "external_blind_used_for_fit": False,
}
joblib.dump(bundle, MODEL_BUNDLE_OUT)

save_parity_plot(
    y_train, train_prediction, y_blind, blind_prediction, PARITY_PNG, PARITY_PDF
)


# ============================================================
# 6. REPORT
# ============================================================

print("=" * 96)
print("TG FINAL60 RANDOM-FOREST EXTERNAL BLIND-SET EVALUATION")
print("=" * 96)
print(f"Final60 descriptors             : {len(selected_features)}")
print(f"Labelled training samples       : {len(y_train)}")
print(f"External blind samples          : {len(y_blind)}")

print("\nExternal blind-set results:")
print(f"R2         = {blind_metrics['r2']:.4f}")
print(f"RMSE       = {blind_metrics['rmse_K']:.3f} K")
print(f"MAE        = {blind_metrics['mae_K']:.3f} K")
print(f"Mean error = {blind_metrics['mean_error_K']:.3f} K")

print("\nLargest absolute external errors:")
print(
    output.nlargest(10, "absolute_error_K")[[
        CODE_COLUMN, "experimental_Tg_K", "predicted_Tg_K", "absolute_error_K"
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}")
)

print("\nSaved:")
for path in [PREDICTIONS_OUT, METRICS_OUT, PARITY_PNG, PARITY_PDF, MODEL_BUNDLE_OUT]:
    print(" -", path)

print("\nDone. Blind Tg labels were used only for post hoc external evaluation.")


