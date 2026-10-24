
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

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

BASE = data_path('')

INPUT_PATH = (
    data_path('DiversityGA_SubsetSize_Comparison/subset_size_comparison.csv')
)

OUTPUT_DIR = (
    data_path('DiversityGA_SubsetSize_Comparison')
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_PNG = (
    data_path('DiversityGA_SubsetSize_Comparison/Performance_plateau_analysis.png')
)

FIGURE_TIFF = (
    data_path('DiversityGA_SubsetSize_Comparison/Performance_plateau_analysis.tiff')
)

FIGURE_PDF = (
    data_path('DiversityGA_SubsetSize_Comparison/Performance_plateau_analysis.pdf')
)


# ============================================================
# 2. SETTINGS
# ============================================================

SELECTED_SUBSET_SIZE = 60

# Now uses cross-validated OOF metrics (held-out set was NOT used
# for dimensionality selection).
METRICS = {
    "OOF_ROC_AUC": "ROC–AUC",
    "OOF_MCC": "MCC",
    "OOF_balanced_accuracy": "Balanced accuracy",
    "OOF_specificity_non_NADES": "Specificity",
}


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file was not found:\n{INPUT_PATH}"
    )

df = pd.read_csv(
    INPUT_PATH
)

required_columns = [
    "subset_size",
    *METRICS.keys(),
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The comparison table is missing columns: "
        f"{missing_columns}"
    )

df = (
    df[required_columns]
    .copy()
    .sort_values(
        by="subset_size"
    )
    .reset_index(drop=True)
)

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

if df[required_columns].isna().any().any():
    raise ValueError(
        "Missing or non-numeric values were detected "
        "in the required columns."
    )

if SELECTED_SUBSET_SIZE not in df["subset_size"].values:
    raise ValueError(
        f"Subset size {SELECTED_SUBSET_SIZE} "
        "is absent from the comparison table."
    )


# ============================================================
# 4. PREPARE SELECTED MODEL VALUES
# ============================================================

selected_row = (
    df.loc[
        df["subset_size"]
        == SELECTED_SUBSET_SIZE
    ]
    .iloc[0]
)

print("=" * 80)
print("PERFORMANCE PLATEAU ANALYSIS  (OOF cross-validation)")
print("=" * 80)

print(
    df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(
    f"\nSelected subset size: "
    f"{SELECTED_SUBSET_SIZE}"
)

for metric_column, metric_label in METRICS.items():
    print(
        f"{metric_label:20s}: "
        f"{selected_row[metric_column]:.4f}"
    )


# ============================================================
# 5. CREATE FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"

fig, ax = plt.subplots(
    figsize=(11, 8)
)

for metric_column, metric_label in METRICS.items():

    ax.plot(
        df["subset_size"],
        df[metric_column],
        marker="o",
        markersize=8,
        linewidth=2.5,
        label=metric_label,
    )

    selected_value = float(
        selected_row[metric_column]
    )

    ax.scatter(
        SELECTED_SUBSET_SIZE,
        selected_value,
        s=150,
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )


# Vertical line indicating the selected subset
ax.axvline(
    x=SELECTED_SUBSET_SIZE,
    linestyle="--",
    linewidth=1.8,
)

# Annotation for the selected model
annotation_y = max(
    float(
        selected_row[metric]
    )
    for metric in METRICS
)

ax.annotate(
    f"Selected {SELECTED_SUBSET_SIZE}-descriptor model",
    xy=(
        SELECTED_SUBSET_SIZE,
        annotation_y,
    ),
    xytext=(
        SELECTED_SUBSET_SIZE - 20,
        annotation_y + 0.030,
    ),
    fontsize=14,
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.5,
    },
)


# ============================================================
# 6. AXIS FORMATTING
# ============================================================

ax.set_xlabel(
    "Number of selected descriptors",
    fontsize=20,
)

ax.set_ylabel(
    "Cross-validated (OOF) performance",
    fontsize=20,
)

ax.set_xticks(
    df["subset_size"]
)

ax.tick_params(
    axis="both",
    labelsize=16,
)

# Adjust according to observed metric range
minimum_value = (
    df[list(METRICS.keys())]
    .min()
    .min()
)

maximum_value = (
    df[list(METRICS.keys())]
    .max()
    .max()
)

lower_limit = max(
    0.0,
    minimum_value - 0.06,
)

upper_limit = min(
    1.0,
    maximum_value + 0.07,
)

ax.set_ylim(
    lower_limit,
    upper_limit,
)

ax.legend(
    frameon=False,
    fontsize=14,
    loc="lower right",
    bbox_to_anchor=(0.95, 0.02),
)

ax.grid(False)

for spine in [
    "top",
    "right",
]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_linewidth(1.3)
ax.spines["bottom"].set_linewidth(1.3)

plt.tight_layout()


# ============================================================
# 7. SAVE FIGURE
# ============================================================

fig.savefig(
    FIGURE_PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_TIFF,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_PDF,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print("\nSaved:")
print(" -", FIGURE_PNG)
print(" -", FIGURE_TIFF)
print(" -", FIGURE_PDF)


