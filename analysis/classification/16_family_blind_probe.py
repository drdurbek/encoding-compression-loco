
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



# ============================================================
# EXPLORATORY FINAL60 FAMILY-ONLY BLIND-SET PROBE
# ============================================================
#
# Purpose
# -------
# For every descriptor family represented among the fixed Final60
# descriptors, fit a separate XGBoost classifier using ONLY the
# descriptors from that family and generate probabilities for the
# blind set.
#
# Important interpretation rule
# -----------------------------
# This is an exploratory diagnostic, NOT a new validation result,
# NOT a replacement for Final60, and NOT a procedure for selecting
# a new threshold or descriptor subset.  Blind labels are never read.
#
# Inputs
# ------
# 1) master_train_features.csv              labelled data, used to fit
# 2) master_blind_features.csv              unlabelled blind data
# 3) GA_candidate_descriptors_1000.csv      candidate-pool family metadata
# 4) Final60_selected_descriptors.csv       the fixed 60 selected descriptors
#
# Outputs
# -------
# Final60_Blind_Family_Probe/
#   Final60_blind_family_only_probabilities_long.csv
#   Final60_blind_family_only_probabilities_wide.csv
#   Final60_blind_family_only_summary.csv
#   Final60_blind_family_only_heatmap.png / .pdf
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.impute import SimpleImputer


# ============================================================
# 1. PATHS AND FROZEN SETTINGS
# ============================================================

BASE = data_path('')

TRAIN_PATH = data_path('master_train_features.csv')
BLIND_PATH = data_path('master_blind_features.csv')

# The two files specified for this experiment.
CANDIDATE_PATH = data_path('GA_candidate_descriptors_1000.csv')
SELECTED_PATH = data_path('Final60_selected_descriptors.csv')

# Optional: if this file exists, its frozen Final60 probabilities are
# shown above the heatmap only as a visual reference.
SELECTED_PATH = (
    data_path('Final60_Internal_Evaluation/Final60_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('Final60_Blind_Family_Probe')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LONG_OUT = (
    data_path('Final60_Blind_Family_Probe/Final60_blind_family_only_probabilities_long.csv')
)
WIDE_OUT = (
    data_path('Final60_Blind_Family_Probe/Final60_blind_family_only_probabilities_wide.csv')
)
SUMMARY_OUT = (
    data_path('Final60_Blind_Family_Probe/Final60_blind_family_only_summary.csv')
)
PLOT_PNG = (
    data_path('Final60_Blind_Family_Probe/Final60_blind_family_only_heatmap.png')
)
PLOT_PDF = (
    data_path('Final60_Blind_Family_Probe/Final60_blind_family_only_heatmap.pdf')
)

TARGET_COLUMN = "activity"
CODE_COLUMN = "Code"
FROZEN_THRESHOLD = 0.42
EXPECTED_FEATURE_COUNT = 60
MODEL_SEED = 82

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

# Same fixed XGBoost configuration used for the Final60 model.
# scale_pos_weight is calculated from the labelled data below.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "random_state": MODEL_SEED,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

plt.rcParams["font.family"] = "Arial"


# ============================================================
# 2. HELPERS
# ============================================================

def find_column(frame, candidates, file_label):
    """Return the first matching column name, ignoring case."""
    normalized = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not find one of {candidates} in {file_label}.\n"
        f"Available columns: {frame.columns.tolist()}"
    )


def numeric_feature_frame(frame, features):
    """Select, numerically coerce, and safely clean a descriptor table."""
    return (
        frame
        .loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def build_family_model(scale_pos_weight):
    return xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )


# ============================================================
# 3. LOAD FILES
# ============================================================

for path, description in [
    (TRAIN_PATH, "Labelled training file"),
    (BLIND_PATH, "Blind feature file"),
    (CANDIDATE_PATH, "GA candidate-descriptor file"),
    (SELECTED_PATH, "Final60 selected-descriptor file"),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}"
        )

train = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

blind = pd.read_csv(
    BLIND_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

candidate_table = pd.read_csv(
    CANDIDATE_PATH,
    low_memory=False,
)

selected_table = pd.read_csv(
    SELECTED_PATH,
    low_memory=False,
)

if TARGET_COLUMN not in train.columns:
    raise ValueError(
        f"'{TARGET_COLUMN}' is missing from {TRAIN_PATH.name}."
    )

if CODE_COLUMN not in blind.columns:
    raise ValueError(
        f"'{CODE_COLUMN}' is missing from {BLIND_PATH.name}."
    )

if blind[CODE_COLUMN].isna().any() or blind[CODE_COLUMN].duplicated().any():
    raise ValueError(
        "Blind-set Code values must be complete and unique."
    )


# ============================================================
# 4. FIXED FINAL60 DESCRIPTORS AND THEIR FAMILIES
# ============================================================

# Robustly recognise the descriptor-name and family columns.
feature_candidates = [
    "feature",
    "descriptor",
    "descriptor_name",
    "selected_feature",
]
family_candidates = [
    "family",
    "descriptor_family",
    "feature_family",
]

selected_feature_col = find_column(
    selected_table,
    feature_candidates,
    SELECTED_PATH.name,
)
selected_family_col = find_column(
    selected_table,
    family_candidates,
    SELECTED_PATH.name,
)

candidate_feature_col = find_column(
    candidate_table,
    feature_candidates,
    CANDIDATE_PATH.name,
)
candidate_family_col = find_column(
    candidate_table,
    family_candidates,
    CANDIDATE_PATH.name,
)

selected_metadata = (
    selected_table[[selected_feature_col, selected_family_col]]
    .rename(
        columns={
            selected_feature_col: "feature",
            selected_family_col: "family_selected",
        }
    )
    .dropna(subset=["feature", "family_selected"])
    .copy()
)

selected_metadata["feature"] = (
    selected_metadata["feature"].astype(str).str.strip()
)
selected_metadata["family_selected"] = (
    selected_metadata["family_selected"].astype(str).str.strip()
)

if selected_metadata["feature"].duplicated().any():
    duplicates = selected_metadata.loc[
        selected_metadata["feature"].duplicated(),
        "feature",
    ].tolist()
    raise ValueError(
        "Duplicate features in Final60_selected_descriptors.csv:\n"
        f"{duplicates}"
    )

candidate_metadata = (
    candidate_table[[candidate_feature_col, candidate_family_col]]
    .rename(
        columns={
            candidate_feature_col: "feature",
            candidate_family_col: "family_candidate",
        }
    )
    .dropna(subset=["feature", "family_candidate"])
    .copy()
)

candidate_metadata["feature"] = (
    candidate_metadata["feature"].astype(str).str.strip()
)
candidate_metadata["family_candidate"] = (
    candidate_metadata["family_candidate"].astype(str).str.strip()
)

if candidate_metadata["feature"].duplicated().any():
    candidate_metadata = candidate_metadata.drop_duplicates(
        subset="feature",
        keep="first",
    )

feature_metadata = selected_metadata.merge(
    candidate_metadata,
    on="feature",
    how="left",
    validate="one_to_one",
)

if feature_metadata["family_candidate"].isna().any():
    missing = feature_metadata.loc[
        feature_metadata["family_candidate"].isna(),
        "feature",
    ].tolist()
    raise ValueError(
        "Some Final60 descriptors are absent from the candidate pool:\n"
        + "\n".join(missing)
    )

inconsistent = feature_metadata.loc[
    feature_metadata["family_selected"]
    != feature_metadata["family_candidate"],
    ["feature", "family_selected", "family_candidate"],
]

if not inconsistent.empty:
    raise ValueError(
        "Family assignments disagree between the selected-descriptor "
        "and GA-candidate files:\n"
        + inconsistent.to_string(index=False)
    )

feature_metadata = feature_metadata.rename(
    columns={"family_selected": "family"}
)[["feature", "family"]]

selected_features = feature_metadata["feature"].tolist()

if len(selected_features) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} Final60 descriptors, found "
        f"{len(selected_features)}."
    )

missing_from_train = [
    feature for feature in selected_features
    if feature not in train.columns
]
missing_from_blind = [
    feature for feature in selected_features
    if feature not in blind.columns
]

if missing_from_train or missing_from_blind:
    message = []
    if missing_from_train:
        message.append(
            "Missing from labelled file:\n"
            + "\n".join(missing_from_train)
        )
    if missing_from_blind:
        message.append(
            "Missing from blind file:\n"
            + "\n".join(missing_from_blind)
        )
    raise ValueError("\n\n".join(message))

family_to_features = {
    family: group["feature"].tolist()
    for family, group in feature_metadata.groupby(
        "family",
        sort=True,
    )
}

family_summary = pd.DataFrame(
    {
        "family": list(family_to_features),
        "n_descriptors": [
            len(features)
            for features in family_to_features.values()
        ],
        "descriptors": [
            "; ".join(features)
            for features in family_to_features.values()
        ],
    }
).sort_values(
    ["n_descriptors", "family"],
    ascending=[False, True],
).reset_index(drop=True)

family_order = family_summary["family"].tolist()


# ============================================================
# 5. LABELLED DATA AND BLIND DESCRIPTORS
# ============================================================

y_series = pd.to_numeric(
    train[TARGET_COLUMN],
    errors="coerce",
)

valid_label_mask = y_series.isin([0, 1])
if not valid_label_mask.all():
    print(
        "Dropped labelled rows with missing/non-binary activity:",
        int((~valid_label_mask).sum()),
    )

train = train.loc[valid_label_mask].copy()
y = y_series.loc[valid_label_mask].astype(np.int8).to_numpy()

if len(np.unique(y)) != 2:
    raise ValueError("The labelled data must contain both classes.")

n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
scale_pos_weight = n_neg / n_pos

X_train_all = numeric_feature_frame(
    train,
    selected_features,
)
X_blind_all = numeric_feature_frame(
    blind,
    selected_features,
)

print("=" * 94)
print("FINAL60 EXPLORATORY BLIND FAMILY-ONLY PROBE")
print("=" * 94)
print(f"Labelled training samples : {len(y)}")
print(f"Class balance 0/1         : {n_neg}/{n_pos}")
print(f"Blind samples              : {len(blind)}")
print(f"Fixed Final60 descriptors  : {len(selected_features)}")
print(f"Descriptor families        : {len(family_order)}")
print(f"Reference threshold        : {FROZEN_THRESHOLD:.2f}")
print("Blind activity is deliberately not read or evaluated.\n")
print(family_summary[["family", "n_descriptors"]].to_string(index=False))


# ============================================================
# 6. FIT ONE FAMILY-ONLY MODEL AND PREDICT THE BLIND SET
# ============================================================

family_probabilities = pd.DataFrame(
    {
        CODE_COLUMN: blind[CODE_COLUMN].astype(str).to_numpy(),
    }
)

long_records = []

for family in family_order:
    family_features = family_to_features[family]

    # Fit imputation and classifier using labelled data only.
    imputer = SimpleImputer(strategy="median")

    X_train_family = imputer.fit_transform(
        X_train_all.loc[:, family_features]
    ).astype(np.float32)

    X_blind_family = imputer.transform(
        X_blind_all.loc[:, family_features]
    ).astype(np.float32)

    family_model = build_family_model(scale_pos_weight)
    family_model.fit(X_train_family, y)

    probability = family_model.predict_proba(
        X_blind_family
    )[:, 1]

    family_probabilities[family] = probability

    for code, value in zip(
        blind[CODE_COLUMN].astype(str),
        probability,
    ):
        long_records.append(
            {
                "Code": code,
                "family": family,
                "n_descriptors": len(family_features),
                "probability_NADES": float(value),
                "predicted_activity_at_reference_tau": int(
                    value >= FROZEN_THRESHOLD
                ),
                "predicted_class_at_reference_tau": (
                    "NADES"
                    if value >= FROZEN_THRESHOLD
                    else "non-NADES"
                ),
            }
        )

    print(
        f"Completed family: {family} "
        f"({len(family_features)} descriptors)"
    )

long_predictions = pd.DataFrame(long_records)


# ============================================================
# 7. OPTIONAL FROZEN FINAL60 REFERENCE PREDICTIONS
# ============================================================

# This is optional and visual only. It does not alter family models.
reference_available = False

if PREDICTIONS_PATH.exists():
    final60_predictions = pd.read_csv(
        PREDICTIONS_PATH,
    )

    required_reference_columns = {
        CODE_COLUMN,
        "probability_NADES",
    }

    if required_reference_columns.issubset(final60_predictions.columns):
        final60_reference = (
            final60_predictions[[CODE_COLUMN, "probability_NADES"]]
            .copy()
        )
        final60_reference[CODE_COLUMN] = (
            final60_reference[CODE_COLUMN].astype(str)
        )

        family_probabilities = family_probabilities.merge(
            final60_reference.rename(
                columns={
                    "probability_NADES": "Final60_frozen_reference",
                }
            ),
            on=CODE_COLUMN,
            how="left",
            validate="one_to_one",
        )

        reference_available = (
            family_probabilities["Final60_frozen_reference"]
            .notna()
            .all()
        )

        if not reference_available:
            family_probabilities = family_probabilities.drop(
                columns="Final60_frozen_reference"
            )
            print(
                "[WARN] Final60 reference codes did not fully match the "
                "blind file; reference panel omitted."
            )


# ============================================================
# 8. SAVE TABLES
# ============================================================

family_summary["mean_probability_NADES"] = [
    float(family_probabilities[family].mean())
    for family in family_order
]
family_summary["sd_probability_NADES"] = [
    float(family_probabilities[family].std(ddof=1))
    for family in family_order
]
family_summary["fraction_at_or_above_reference_tau"] = [
    float(
        (family_probabilities[family] >= FROZEN_THRESHOLD).mean()
    )
    for family in family_order
]

if reference_available:
    family_summary["mean_abs_difference_vs_Final60"] = [
        float(
            np.mean(
                np.abs(
                    family_probabilities[family]
                    - family_probabilities["Final60_frozen_reference"]
                )
            )
        )
        for family in family_order
    ]

long_predictions.to_csv(LONG_OUT, index=False)
family_probabilities.to_csv(WIDE_OUT, index=False)
family_summary.to_csv(SUMMARY_OUT, index=False)


# ============================================================
# 9. PLOT: BLIND FAMILY PROBABILITY MAP
# ============================================================

heat_matrix = (
    family_probabilities
    .set_index(CODE_COLUMN)
    .loc[:, family_order]
    .T
)

n_families = len(family_order)
codes = family_probabilities[CODE_COLUMN].astype(str).tolist()
x_position = np.arange(len(codes))

fig, ax_heatmap = plt.subplots(
    figsize=(max(17, 0.45 * len(codes)), max(2.6, 0.55 * n_families + 1.8)),
)

image = ax_heatmap.imshow(
    heat_matrix.to_numpy(dtype=float),
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    aspect="auto",
    interpolation="nearest",
)

ax_heatmap.set_yticks(np.arange(n_families))
ax_heatmap.set_yticklabels(
    [
        f"{family} (n={len(family_to_features[family])})"
        for family in family_order
    ],
    fontsize=11,
)

ax_heatmap.set_xticks(x_position)
ax_heatmap.set_xticklabels(codes, rotation=0, fontsize=10)

ax_heatmap.set_xlabel(
    "Blind-set sample code",
    fontsize=12,
)
ax_heatmap.set_ylabel(
    "Final60 descriptor family",
    fontsize=12,
)

ax_heatmap.set_title(
    "Exploratory blind-set predictions from individual Final60 descriptor families",
    fontsize=14,
    pad=12,
)

# Fine white separators between cells
ax_heatmap.set_xticks(
    np.arange(-0.5, len(codes), 1),
    minor=True,
)
ax_heatmap.set_yticks(
    np.arange(-0.5, n_families, 1),
    minor=True,
)
ax_heatmap.grid(
    which="minor",
    color="white",
    linewidth=0.45,
)
ax_heatmap.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

colorbar = fig.colorbar(
    image,
    ax=ax_heatmap,
    pad=0.015,
)
colorbar.set_label(
    "Family-only predicted probability of NADES",
    fontsize=11,
)

ax_heatmap.spines["top"].set_visible(False)
ax_heatmap.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    PLOT_PNG,
    dpi=600,
    bbox_inches="tight",
)
fig.savefig(
    PLOT_PDF,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "=" * 94)
print("FAMILY-ONLY BLIND PROBE COMPLETED")
print("=" * 94)
print(
    family_summary[
        [
            "family",
            "n_descriptors",
            "mean_probability_NADES",
            "sd_probability_NADES",
            "fraction_at_or_above_reference_tau",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nSaved:")
for path in [LONG_OUT, WIDE_OUT, SUMMARY_OUT, PLOT_PNG, PLOT_PDF]:
    print(" -", path)

print(
    "\nInterpretation: family-only probabilities are exploratory. "
    "Do not use this blind-set output to change the Final60 descriptors, "
    "threshold, or reported external-validation result."
)

