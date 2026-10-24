
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
# BLIND PREDICTIONS — COMPONENT_B FAMILY ONLY
# Requires:
#   family_probabilities
#   FROZEN_THRESHOLD
#   OUTPUT_DIR
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

FAMILY_TO_PLOT = "Component_B"

if FAMILY_TO_PLOT not in family_probabilities.columns:
    raise ValueError(
        f"Family '{FAMILY_TO_PLOT}' was not found.\n"
        f"Available families: "
        f"{[c for c in family_probabilities.columns if c not in ['Code', 'Final60_frozen_reference']]}"
    )

plot_df = family_probabilities[
    ["Code", FAMILY_TO_PLOT]
].copy()

plot_df["Code_numeric"] = pd.to_numeric(
    plot_df["Code"],
    errors="coerce",
)

plot_df = (
    plot_df
    .sort_values(["Code_numeric", "Code"])
    .reset_index(drop=True)
)

probabilities = plot_df[
    FAMILY_TO_PLOT
].to_numpy(dtype=float)

positions = np.arange(len(plot_df))

colors = np.where(
    probabilities >= FROZEN_THRESHOLD,
    "#91D1C2",   # NADES at frozen threshold
    "#F39B7F",   # non-NADES at frozen threshold
)

fig, ax = plt.subplots(
    figsize=(max(12, 0.47 * len(plot_df)), 5.5)
)

ax.bar(
    positions,
    probabilities,
    color=colors,
    edgecolor="white",
    linewidth=0.8,
)

ax.axhline(
    FROZEN_THRESHOLD,
    color="#2A6FBB",
    linestyle="--",
    linewidth=1.6,
    zorder=4,
)

ax.text(
    len(plot_df) + 0.55,
    FROZEN_THRESHOLD + 0.02,
    rf"$\tau = {FROZEN_THRESHOLD:.2f}$",
    ha="right",
    va="bottom",
    fontsize=12,
    color="#2A6FBB",
)

for position, probability in zip(positions, probabilities):

    label = (
        f"{probability:.3f}"
        if abs(probability - FROZEN_THRESHOLD) < 0.03
        else f"{probability:.2f}"
    )

    label_y = probability + 0.025

    if abs(label_y - FROZEN_THRESHOLD) < 0.05:
        label_y = FROZEN_THRESHOLD + 0.04

    ax.text(
        position,
        min(label_y, 1.035),
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        zorder=5,
    )

ax.set_xlabel(
    "Blind-set sample code",
    fontsize=14,
)

ax.set_ylabel(
    "Predicted probability of NADES based on B component",
    fontsize=14,
)

ax.set_xticks(positions)
ax.set_xticklabels(
    plot_df["Code"].astype(str),
    fontsize=11,
)

ax.set_ylim(0.0, 1.08)

ax.tick_params(
    axis="y",
    labelsize=12,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#B5B5B5")
ax.spines["bottom"].set_color("#B5B5B5")

ax.grid(False)

fig.tight_layout()

component_b_png = (
    OUTPUT_DIR
    / "Final60_blind_Component_B_family_probabilities.png"
)

component_b_pdf = (
    OUTPUT_DIR
    / "Final60_blind_Component_B_family_probabilities.pdf"
)

fig.savefig(
    component_b_png,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    component_b_pdf,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print("Saved:")
print(" -", component_b_png)
print(" -", component_b_pdf)


