
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


BASE = data_path('TG')

OUTPUT_DIR = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison')
)

INPUT_PATH = data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_all_GA_runs.csv')

SUMMARY_OUT = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_subset_size_mean_sd_summary.csv')
)

FIGURE_PNG = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_mean_OOF_RMSE_vs_subset_size.png')
)

FIGURE_PDF = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_mean_OOF_RMSE_vs_subset_size.pdf')
)

FIGURE_TIFF = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_mean_OOF_RMSE_vs_subset_size.tiff')
)

SELECTED_SUBSET_SIZE = 60

plt.rcParams["font.family"] = "Arial"


# ============================================================
# LOAD AND SUMMARIZE ALL INDEPENDENT GA RUNS
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "subset_size",
    "oof_r2",
    "oof_rmse",
    "oof_mae",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

summary = (
    df
    .groupby("subset_size", as_index=False)
    .agg(
        n_runs=("oof_rmse", "count"),
        mean_oof_r2=("oof_r2", "mean"),
        sd_oof_r2=("oof_r2", "std"),
        mean_oof_rmse_K=("oof_rmse", "mean"),
        sd_oof_rmse_K=("oof_rmse", "std"),
        mean_oof_mae_K=("oof_mae", "mean"),
        sd_oof_mae_K=("oof_mae", "std"),
    )
    .sort_values("subset_size")
    .reset_index(drop=True)
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
)

selected = (
    summary.loc[
        summary["subset_size"] == SELECTED_SUBSET_SIZE
    ]
    .iloc[0]
)

print("=" * 90)
print("TG RF GA: MEAN GROUPED-OOF PERFORMANCE ACROSS INDEPENDENT RUNS")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(
    f"\nFinal60 mean OOF RMSE = "
    f"{selected['mean_oof_rmse_K']:.3f} ± "
    f"{selected['sd_oof_rmse_K']:.3f} K"
)


# ============================================================
# PLOT: PRIMARY SELECTION METRIC
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(
    summary["subset_size"],
    summary["mean_oof_rmse_K"],
    yerr=summary["sd_oof_rmse_K"],
    fmt="o-",
    color="#E76F51",
    ecolor="#E76F51",
    elinewidth=1.5,
    capsize=5,
    markersize=8,
    linewidth=2.5,
    label="Mean grouped OOF RMSE ± SD",
)

ax.scatter(
    SELECTED_SUBSET_SIZE,
    selected["mean_oof_rmse_K"],
    s=160,
    color="#E76F51",
    edgecolors="black",
    linewidths=1.5,
    zorder=5,
)

ax.axvline(
    SELECTED_SUBSET_SIZE,
    linestyle="--",
    linewidth=1.5,
    color="#666666",
)

ax.annotate(
    (
        "Final60\n"
        f"RMSE = {selected['mean_oof_rmse_K']:.2f} ± "
        f"{selected['sd_oof_rmse_K']:.2f} K"
    ),
    xy=(
        SELECTED_SUBSET_SIZE,
        selected["mean_oof_rmse_K"],
    ),
    xytext=(45, selected["mean_oof_rmse_K"] - 0.41),
    fontsize=13,
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.2,
        "color": "#444444",
    },
)

ax.set_xlabel(
    "Number of selected descriptors",
    fontsize=16,
)

ax.set_ylabel(
    "Mean grouped OOF RMSE (K)",
    fontsize=16,
)

ax.set_xticks(summary["subset_size"])
ax.tick_params(axis="both", labelsize=13)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.grid(False)

ax.legend(
    frameon=False,
    fontsize=12,
    loc="upper left",
)

fig.tight_layout()

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
print(" -", SUMMARY_OUT)
print(" -", FIGURE_PNG)
print(" -", FIGURE_TIFF)
print(" -", FIGURE_PDF)

