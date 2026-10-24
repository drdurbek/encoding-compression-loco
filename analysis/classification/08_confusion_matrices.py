
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
# CONFUSION MATRICES — TRAIN AND TEST
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import numpy as np


CONFUSION_MATRIX_OUT = (
    OUTPUT_DIR / "Final40_confusion_matrices_train_test.png"
)

POSITIVE_COLOR = "#91D1C2"   # NADES / true positive
NEGATIVE_COLOR = "#F39B7F"   # non-NADES / true negative
ERROR_COLOR = "#F1F1F1"      # false positive / false negative


# ------------------------------------------------------------
# 1. Predictions using the fixed threshold
# ------------------------------------------------------------

train_prediction = (
    train_probability >= THRESHOLD
).astype(np.int8)

test_prediction = (
    test_probability >= THRESHOLD
).astype(np.int8)


# ------------------------------------------------------------
# 2. Confusion matrices and reports
# ------------------------------------------------------------

cm_train = confusion_matrix(
    y_train,
    train_prediction,
    labels=[0, 1],
)

cm_test = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1],
)

print("\n================ TRAIN CONFUSION MATRIX ================")
print(cm_train)

print(
    classification_report(
        y_train,
        train_prediction,
        labels=[0, 1],
        target_names=["non-NADES", "NADES"],
        digits=4,
        zero_division=0,
    )
)

print("\n================ TEST CONFUSION MATRIX =================")
print(cm_test)

print(
    classification_report(
        y_test,
        test_prediction,
        labels=[0, 1],
        target_names=["non-NADES", "NADES"],
        digits=4,
        zero_division=0,
    )
)


# ------------------------------------------------------------
# 3. Plot helper
# ------------------------------------------------------------

class_labels = ["non-NADES", "NADES"]


def plot_confusion_matrix(ax, cm, title):
    # Rows = true class; columns = predicted class
    # TN = salmon, TP = mint, errors = light gray
    cell_colors = np.array(
        [
            [to_rgba(NEGATIVE_COLOR), to_rgba(ERROR_COLOR)],
            [to_rgba(ERROR_COLOR), to_rgba(POSITIVE_COLOR)],
        ]
    )

    ax.imshow(
        cell_colors,
        interpolation="nearest",
    )

    ax.set_title(
        title,
        fontsize=20,
        pad=12,
    )
    ax.set_xlabel(
        "Predicted class",
        fontsize=18,
    )
    ax.set_ylabel(
        "True class",
        fontsize=18,
    )

    ax.set_xticks(
        [0, 1],
        labels=class_labels,
        fontsize=15,
    )
    ax.set_yticks(
        [0, 1],
        labels=class_labels,
        fontsize=15,
    )

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
                fontsize=26,
                fontweight="bold",
                color="black",
            )

    # White borders between the four cells
    ax.set_xticks(
        np.arange(-0.5, 2, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, 2, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color="white",
        linewidth=3,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


# ------------------------------------------------------------
# 4. Plot train and test matrices
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6),
)

plot_confusion_matrix(
    axes[0],
    cm_train,
    "Train Confusion Matrix",
)

plot_confusion_matrix(
    axes[1],
    cm_test,
    "Test Confusion Matrix",
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

print(f"Saved: {CONFUSION_MATRIX_OUT}")

