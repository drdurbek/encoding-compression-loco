
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
# HELD-OUT TEST ROC + PR CURVES WITH OPERATING POINT
# Fixed operating threshold: tau = THRESHOLD
#
# Uses:
#   y_test
#   test_probability
#   THRESHOLD
#   OUTPUT_DIR
#
# Saves:
#   Final60_ROC_PR_Test_thr064.png
#   Final60_ROC_PR_Test_thr064.pdf
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ============================================================
# 1. STYLE
# ============================================================

plt.rcParams["font.family"] = "Arial"

FS = 14
LW_MAIN = 2.0
LW_DASH = 1.5
DOT_SIZE = 60

CURVE_COLOR = "#00A087FF"
TAU_COLOR = "#4C78A8"
REFERENCE_COLOR = "#999999"

# The same fixed threshold used by the classification model.
thr_use = float(THRESHOLD)

# ============================================================
# 2. HELD-OUT TEST DATA
# ============================================================

y_true = np.asarray(
    y_test,
    dtype=np.int8,
)

y_proba = np.asarray(
    test_probability,
    dtype=float,
)

if len(y_true) != len(y_proba):
    raise ValueError(
        "y_test and test_probability have different lengths."
    )

if np.isnan(y_proba).any():
    raise ValueError(
        "test_probability contains missing values."
    )

if np.any(
    (y_proba < 0.0) | (y_proba > 1.0)
):
    raise ValueError(
        "Predicted probabilities must be between 0 and 1."
    )

# ============================================================
# 3. ROC AND PRECISION–RECALL CURVES
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_proba,
)

precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_proba,
    )
)

roc_auc = roc_auc_score(
    y_true,
    y_proba,
)

average_precision = average_precision_score(
    y_true,
    y_proba,
)

# Baseline precision for the PR curve:
# fraction of positive samples in the test set.
positive_prevalence = float(
    np.mean(y_true == 1)
)

# ============================================================
# 4. CLASSIFICATION AT THE FIXED THRESHOLD
# ============================================================

y_pred = (
    y_proba >= thr_use
).astype(np.int8)

precision_at_threshold = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall_at_threshold = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1_at_threshold = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

specificity_at_threshold = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)

fpr_at_threshold = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else np.nan
)

# ============================================================
# 5. EXACT OPERATING POINT
# ============================================================

# ROC operating point:
# x = false-positive rate
# y = true-positive rate
roc_operating_x = fpr_at_threshold
roc_operating_y = recall_at_threshold

# PR operating point:
# x = recall
# y = precision
pr_operating_x = recall_at_threshold
pr_operating_y = precision_at_threshold

# ============================================================
# 6. CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6),
)

# ------------------------------------------------------------
# Panel A: ROC curve
# ------------------------------------------------------------

axes[0].plot(
    fpr,
    tpr,
    linewidth=LW_MAIN,
    color=CURVE_COLOR,
    label=(
        f"Test ROC "
        f"(AUC = {roc_auc:.3f})"
    ),
)

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=LW_DASH,
    color=REFERENCE_COLOR,
)

axes[0].scatter(
    roc_operating_x,
    roc_operating_y,
    s=DOT_SIZE,
    color=TAU_COLOR,
    edgecolor="white",
    linewidth=0.6,
    zorder=5,
)

axes[0].annotate(
    rf"$\tau = {thr_use:.2f}$",
    xy=(
        roc_operating_x,
        roc_operating_y,
    ),
    xytext=(-8, 5),
    textcoords="offset points",
    horizontalalignment="right",
    verticalalignment="top",
    fontsize=12,
    color=TAU_COLOR,
)

axes[0].set_xlabel(
    "False Positive Rate",
    fontsize=FS,
)

axes[0].set_ylabel(
    "True Positive Rate",
    fontsize=FS,
)

axes[0].tick_params(
    axis="both",
    labelsize=12,
)

axes[0].set_xlim(
    0.0,
    1.0,
)

axes[0].set_ylim(
    0.0,
    1.02,
)

axes[0].legend(
    loc="lower right",
    frameon=False,
    fontsize=12,
)

axes[0].grid(False)

axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].spines["left"].set_visible(True)
axes[0].spines["bottom"].set_visible(True)

# Optional panel label.
axes[0].text(
    -0.14,
    1.03,
    "",
    transform=axes[0].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

# ------------------------------------------------------------
# Panel B: Precision–Recall curve
# ------------------------------------------------------------

axes[1].plot(
    recall,
    precision,
    linewidth=LW_MAIN,
    color=CURVE_COLOR,
    label=(
        f"Test PR "
        f"(AP = {average_precision:.3f})"
    ),
)

# Random-classifier baseline for PR curve.
axes[1].axhline(
    y=positive_prevalence,
    linestyle="--",
    linewidth=LW_DASH,
    color=REFERENCE_COLOR,
    label=(
        f"Baseline = "
        f"{positive_prevalence:.3f}"
    ),
)

axes[1].scatter(
    pr_operating_x,
    pr_operating_y,
    s=DOT_SIZE,
    color=TAU_COLOR,
    edgecolor="white",
    linewidth=0.6,
    zorder=2,
)

axes[1].annotate(
    rf"$\tau = {thr_use:.2f}$",
    xy=(
        pr_operating_x,
        pr_operating_y,
    ),
    xytext=(5, 8),
    textcoords="offset points",
    horizontalalignment="right",
    verticalalignment="bottom",
    fontsize=12,
    color=TAU_COLOR,
)

axes[1].set_xlabel(
    "Recall",
    fontsize=FS,
)

axes[1].set_ylabel(
    "Precision",
    fontsize=FS,
)

axes[1].tick_params(
    axis="both",
    labelsize=12,
)

axes[1].set_xlim(
    0.0,
    1.0,
)

axes[1].set_ylim(
    0.0,
    1.02,
)

axes[1].legend(
    loc="lower left",
    frameon=False,
    fontsize=12,
)

axes[1].grid(False)

axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].spines["left"].set_visible(True)
axes[1].spines["bottom"].set_visible(True)

# Optional panel label.
axes[1].text(
    -0.14,
    1.03,
    "",
    transform=axes[1].transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

plt.tight_layout()

# ============================================================
# 7. SAVE FIGURE
# ============================================================

threshold_label = (
    f"{thr_use:.2f}".replace(".", "")
)

roc_pr_png = (
    OUTPUT_DIR
    / f"Final60_ROC_PR_Test_thr{threshold_label}.png"
)

roc_pr_pdf = (
    OUTPUT_DIR
    / f"Final60_ROC_PR_Test_thr{threshold_label}.pdf"
)

plt.savefig(
    roc_pr_png,
    dpi=600,
    bbox_inches="tight",
)

plt.savefig(
    roc_pr_pdf,
    bbox_inches="tight",
)

plt.show()
plt.close()

# ============================================================
# 8. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("HELD-OUT TEST ROC AND PRECISION–RECALL RESULTS")
print("=" * 90)

print(
    f"Test ROC-AUC             : "
    f"{roc_auc:.4f}"
)

print(
    f"Test average precision   : "
    f"{average_precision:.4f}"
)

print(
    f"PR baseline prevalence   : "
    f"{positive_prevalence:.4f}"
)

print(
    f"\nTest results at tau = "
    f"{thr_use:.2f}"
)

print(
    f"Precision NADES          : "
    f"{precision_at_threshold:.4f}"
)

print(
    f"Recall/TPR NADES         : "
    f"{recall_at_threshold:.4f}"
)

print(
    f"F1 NADES                 : "
    f"{f1_at_threshold:.4f}"
)

print(
    f"Specificity/TNR          : "
    f"{specificity_at_threshold:.4f}"
)

print(
    f"False Positive Rate      : "
    f"{fpr_at_threshold:.4f}"
)

print("\nTest confusion matrix:")
print(cm)

print("\nSaved:")
print(" -", roc_pr_png)
print(" -", roc_pr_pdf)

