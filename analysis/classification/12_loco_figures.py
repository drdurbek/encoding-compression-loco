
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
# FINAL60 LOCO FIGURES
# Reconstructs component curves from saved LOCO predictions
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE = data_path('')

OUTPUT_DIR = data_path('Final60_LOCO')

RESULTS_PATH = (
    data_path('Final60_LOCO/Final60_LOCO_component_results.csv')
)

PREDICTIONS_PATH = (
    data_path('Final60_LOCO/Final60_LOCO_sample_predictions.csv')
)

TOP_K_CURVES = 12

FS_LABEL = 16
FS_TICK = 12
FS_LEGEND = 11

BAR_COLOR = "#298c8c"
REFERENCE_COLOR = "#666666"

plt.rcParams["font.family"] = "Arial"


def style_axis(ax):
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#B5B5B5")
    ax.spines["bottom"].set_color("#B5B5B5")

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def save_figure(fig, filename_stem):
    png_path = OUTPUT_DIR / f"{filename_stem}.png"
    pdf_path = OUTPUT_DIR / f"{filename_stem}.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print("Saved:")
    print(" -", png_path)
    print(" -", pdf_path)


# ============================================================
# 1. LOAD FINAL60 LOCO RESULTS
# ============================================================

results_df = pd.read_csv(RESULTS_PATH)
predictions_df = pd.read_csv(PREDICTIONS_PATH)

plot_df = results_df.dropna(
    subset=[
        "ROC_AUC",
        "PR_AUC_NADES",
    ]
).copy()

# ============================================================
# 2. RECONSTRUCT ROC AND PR CURVES FROM SAVED PREDICTIONS
# ============================================================

roc_points = {}
pr_points = {}

for component, component_df in predictions_df.groupby(
    "held_out_component"
):
    y_component = component_df[
        "true_activity"
    ].to_numpy(
        dtype=int
    )

    probability_component = component_df[
        "probability_NADES"
    ].to_numpy(
        dtype=float
    )

    if np.unique(y_component).size != 2:
        continue

    fpr, tpr, _ = roc_curve(
        y_component,
        probability_component,
    )

    precision, recall, _ = precision_recall_curve(
        y_component,
        probability_component,
    )

    roc_points[component] = (
        fpr,
        tpr,
    )

    pr_points[component] = (
        precision,
        recall,
    )

print(
    f"Components with reconstructed ROC/PR curves: "
    f"{len(roc_points)}"
)

# ============================================================
# 3. ROC-AUC BAR PLOT
# ============================================================

roc_bar_df = (
    plot_df
    .sort_values(
        "ROC_AUC",
        ascending=True,
    )
    .reset_index(drop=True)
)

fig_height = max(
    7.0,
    0.34 * len(roc_bar_df),
)

fig, ax = plt.subplots(
    figsize=(9, fig_height)
)

positions = np.arange(
    len(roc_bar_df)
)

ax.barh(
    positions,
    roc_bar_df["ROC_AUC"],
    color=BAR_COLOR,
)

ax.axvline(
    roc_bar_df["ROC_AUC"].mean(),
    linestyle="--",
    color=REFERENCE_COLOR,
    linewidth=1.3,
    label=(
        f"Mean = "
        f"{roc_bar_df['ROC_AUC'].mean():.3f}"
    ),
)

ax.axvline(
    0.5,
    linestyle=":",
    color=REFERENCE_COLOR,
    linewidth=1.2,
    label="Random ranking = 0.5",
)

ax.set_yticks(positions)

ax.set_yticklabels(
    roc_bar_df["component"],
    fontsize=FS_TICK,
)

ax.set_xlabel(
    "ROC–AUC",
    fontsize=FS_LABEL,
)

ax.set_ylabel(
    "Held-out component",
    fontsize=FS_LABEL,
)

ax.set_xlim(0.0, 1.02)

ax.legend(
    frameon=False,
    fontsize=FS_LEGEND,
    loc="lower right",
)

style_axis(ax)

fig.tight_layout()

save_figure(
    fig,
    "Final60_LOCO_ROC_AUC_per_component",
)

plt.show()
plt.close(fig)

# ============================================================
# 4. PR-AUC BAR PLOT WITH COMPONENT PREVALENCE
# ============================================================

pr_bar_df = (
    plot_df
    .sort_values(
        "PR_AUC_NADES",
        ascending=True,
    )
    .reset_index(drop=True)
)

fig_height = max(
    7.0,
    0.34 * len(pr_bar_df),
)

fig, ax = plt.subplots(
    figsize=(9, fig_height)
)

positions = np.arange(
    len(pr_bar_df)
)

ax.barh(
    positions,
    pr_bar_df["PR_AUC_NADES"],
    color=BAR_COLOR,
    label="Observed AP",
)

ax.scatter(
    pr_bar_df["PR_baseline_NADES"],
    positions,
    marker="|",
    s=140,
    linewidths=2,
    color="orange",
    label="NADES prevalence baseline",
    zorder=3,
)

ax.axvline(
    pr_bar_df["PR_AUC_NADES"].mean(),
    color=REFERENCE_COLOR,
    linestyle="--",
    linewidth=1.3,
    label=(
        f"Mean AP = "
        f"{pr_bar_df['PR_AUC_NADES'].mean():.3f}"
    ),
)

ax.set_yticks(positions)

ax.set_yticklabels(
    pr_bar_df["component"],
    fontsize=FS_TICK,
)

ax.set_xlabel(
    "Average precision for NADES",
    fontsize=FS_LABEL,
)

ax.set_ylabel(
    "Held-out component",
    fontsize=FS_LABEL,
)

ax.set_xlim(0.0, 1.02)

ax.legend(
    frameon=False,
    fontsize=FS_LEGEND,
    loc="lower right",
)

style_axis(ax)

fig.tight_layout()

save_figure(
    fig,
    "Final60_LOCO_PR_AUC_per_component",
)

plt.show()
plt.close(fig)

# ============================================================
# 5. TOP ROC CURVES
# ============================================================

top_components = (
    plot_df
    .sort_values(
        "n_test",
        ascending=False,
    )["component"]
    .head(TOP_K_CURVES)
    .tolist()
)

top_components_roc = [
    component
    for component in top_components
    if component in roc_points
]

if top_components_roc:

    fig, ax = plt.subplots(
        figsize=(10, 10),
    )

    colors = plt.cm.tab20(
        np.linspace(
            0.0,
            1.0,
            len(top_components_roc),
        )
    )

    for component, color in zip(
        top_components_roc,
        colors,
    ):
        fpr, tpr = roc_points[component]

        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=1.8,
            label=component,
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        color="#999999",
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=20,
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=20,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)

    ax.legend(
        ncol=3,
        frameon=True,
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.59, 0.01),
        borderaxespad=0.1,
    )

    style_axis(ax)

    ax.tick_params(
        axis="both",
        labelsize=16,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Final60_LOCO_ROC_curves_top_components",
    )

    plt.show()
    plt.close(fig)

# ============================================================
# 6. TOP PR CURVES
# ============================================================

top_components_pr = [
    component
    for component in top_components
    if component in pr_points
]

if top_components_pr:

    fig, ax = plt.subplots(
        figsize=(10, 10),
    )

    colors = plt.cm.tab20(
        np.linspace(
            0.0,
            1.0,
            len(top_components_pr),
        )
    )

    for component, color in zip(
        top_components_pr,
        colors,
    ):
        precision, recall = pr_points[component]

        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=1.8,
            label=component,
        )

    ax.set_xlabel(
        "Recall",
        fontsize=20,
    )

    ax.set_ylabel(
        "Precision",
        fontsize=20,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)

    ax.legend(
        ncol=3,
        frameon=True,
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.59, 0.01),
        borderaxespad=0.1,
    )

    style_axis(ax)

    ax.tick_params(
        axis="both",
        labelsize=16,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Final60_LOCO_PR_curves_top_components",
    )

    plt.show()
    plt.close(fig)


