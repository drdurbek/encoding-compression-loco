
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
# BLIND-SET PREDICTED NADES PROBABILITIES
# Final 60-descriptor XGBoost model
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE = data_path('')

PREDICTIONS_PATH = (
    data_path('Final60_blind_predictions.csv')
)

PLOT_PNG = (
    data_path('Final60_blind_prediction_probabilities.png')
)

PLOT_PDF = (
    data_path('Final60_blind_prediction_probabilities.pdf')
)

THRESHOLD = 0.42

POSITIVE_COLOR = "#91D1C2"
NEGATIVE_COLOR = "#F39B7F"
THRESHOLD_COLOR = "#2A6FBB"

plt.rcParams["font.family"] = "Arial"

# ============================================================
# 2. LOAD AND SORT PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTIONS_PATH,
)

required_columns = {
    "Code",
    "probability_NADES",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "Missing columns in blind prediction file: "
        f"{sorted(missing_columns)}"
    )

df["Code"] = df["Code"].astype(str)

df["Code_numeric"] = pd.to_numeric(
    df["Code"],
    errors="coerce",
)

if df["Code_numeric"].notna().all():
    df = df.sort_values(
        "Code_numeric"
    )
else:
    df = df.sort_values(
        "Code"
    )

df = df.reset_index(
    drop=True
)

probabilities = df[
    "probability_NADES"
].to_numpy(
    dtype=float
)

colors = np.where(
    probabilities >= THRESHOLD,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
)

# ============================================================
# 3. CREATE FIGURE
# ============================================================

fig_width = max(
    7,
    0.50 * len(df),
)

fig, ax = plt.subplots(
    figsize=(fig_width, 6),
)

positions = np.arange(
    len(df)
)

ax.bar(
    positions,
    probabilities,
    color=colors,
    edgecolor="white",
    linewidth=0.8,
)

# Fixed operating threshold.
ax.axhline(
    y=THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    color="#2A6FBB",
    alpha=1.0,
    zorder=5,
)

ax.text(
    len(df) + 1.15,
    THRESHOLD + 0.015,
    rf"$\tau = {THRESHOLD:.2f}$",
    ha="right",
    va="bottom",
    fontsize=13,
    fontweight="bold",
    color="#2A6FBB",
    zorder=5,
    bbox=dict(
        boxstyle="round,pad=0.18",
        facecolor="white",
        edgecolor="none",
        alpha=0.75,
    ),
)

for position, probability in zip(
    positions,
    probabilities,
):
    if abs(probability - THRESHOLD) < 0.005:
        label = f"{probability:.4f}"
    elif abs(probability - THRESHOLD) < 0.025:
        label = f"{probability:.3f}"
    else:
        label = f"{probability:.2f}"

    label_y = probability + 0.02

    if abs(label_y - THRESHOLD) < 0.05:
        label_y = THRESHOLD + 0.035

    ax.text(
        position,
        min(label_y, 1.035),
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        zorder=6,
    )

# ============================================================
# 4. STYLE
# ============================================================

ax.set_xlabel(
    "Blind-set sample code",
    fontsize=12,
)

ax.set_ylabel(
    "Predicted probability of NADES",
    fontsize=12,
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    df["Code"],
    fontsize=12,
)

ax.set_ylim(
    0.0,
    1.08,
)

ax.tick_params(
    axis="y",
    labelsize=12,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#B5B5B5")
ax.spines["bottom"].set_color("#B5B5B5")

ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.grid(False)

fig.tight_layout()

# ============================================================
# 5. SAVE
# ============================================================

plt.savefig(
    PLOT_PNG,
    dpi=600,
    bbox_inches="tight",
)

plt.savefig(
    PLOT_PDF,
    bbox_inches="tight",
)

plt.show()
plt.close()

print("Saved:")
print(" -", PLOT_PNG)
print(" -", PLOT_PDF)

