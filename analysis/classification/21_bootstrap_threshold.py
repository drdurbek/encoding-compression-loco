
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
# HELD-OUT TEST: STRATIFIED BOOTSTRAP CI AT FIXED THRESHOLD
# Final60 XGBoost model
#
# Uses:
#   y_test
#   test_probability
#   THRESHOLD
#   OUTPUT_DIR
#
# Saves:
#   Final60_heldout_threshold_metrics_bootstrap_CI.csv
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# 1. SETTINGS
# ============================================================

N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95

thr_use = float(THRESHOLD)  # Fixed before held-out evaluation

# ============================================================
# 2. HELPER: METRICS AT FIXED THRESHOLD
# ============================================================

def threshold_metrics(
    y_true,
    y_score,
    threshold,
):
    y_pred = (
        y_score >= threshold
    ).astype(
        np.int8
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    balanced_accuracy = (
        0.5 * (recall + specificity)
        if np.isfinite(specificity)
        else np.nan
    )

    return {
        "Precision": float(precision),
        "Recall_TPR": float(recall),
        "F1": float(f1),
        "Specificity_TNR": float(specificity),
        "False_positive_rate": float(false_positive_rate),
        "Balanced_accuracy": float(balanced_accuracy),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

# ============================================================
# 3. STRATIFIED BOOTSTRAP CI
# ============================================================

def bootstrap_ci_threshold_metrics_stratified(
    y_true,
    y_score,
    threshold,
    n_boot=5000,
    seed=42,
    ci=0.95,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score have different lengths."
        )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError(
            "y_true must contain binary labels 0 and 1."
        )

    if np.isnan(y_score).any():
        raise ValueError(
            "y_score contains missing values."
        )

    if np.any(
        (y_score < 0.0)
        | (y_score > 1.0)
    ):
        raise ValueError(
            "y_score must contain probabilities between 0 and 1."
        )

    index_nades = np.where(
        y_true == 1
    )[0]

    index_non_nades = np.where(
        y_true == 0
    )[0]

    n_nades = len(
        index_nades
    )

    n_non_nades = len(
        index_non_nades
    )

    if n_nades == 0 or n_non_nades == 0:
        raise ValueError(
            "Both NADES and non-NADES samples are required."
        )

    point_metrics = threshold_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
    )

    metric_names = [
        "Precision",
        "Recall_TPR",
        "F1",
        "Specificity_TNR",
        "False_positive_rate",
        "Balanced_accuracy",
    ]

    bootstrap_values = {
        metric: np.empty(
            n_boot,
            dtype=float,
        )
        for metric in metric_names
    }

    rng = np.random.default_rng(
        seed
    )

    for bootstrap_index in range(n_boot):

        sampled_nades = rng.choice(
            index_nades,
            size=n_nades,
            replace=True,
        )

        sampled_non_nades = rng.choice(
            index_non_nades,
            size=n_non_nades,
            replace=True,
        )

        sampled_index = np.concatenate(
            [
                sampled_nades,
                sampled_non_nades,
            ]
        )

        bootstrap_metrics = threshold_metrics(
            y_true=y_true[sampled_index],
            y_score=y_score[sampled_index],
            threshold=threshold,
        )

        for metric in metric_names:
            bootstrap_values[metric][
                bootstrap_index
            ] = bootstrap_metrics[metric]

    alpha = (
        1.0 - ci
    ) / 2.0

    lower_percentile = 100.0 * alpha
    upper_percentile = 100.0 * (1.0 - alpha)

    result = {
        "threshold": float(threshold),
        "n_samples": int(
            n_nades + n_non_nades
        ),
        "n_NADES": int(n_nades),
        "n_non_NADES": int(n_non_nades),
        "n_bootstrap": int(n_boot),
        "confidence_level": float(ci),
        **point_metrics,
    }

    for metric in metric_names:
        result[f"{metric}_CI_low"] = float(
            np.percentile(
                bootstrap_values[metric],
                lower_percentile,
            )
        )

        result[f"{metric}_CI_high"] = float(
            np.percentile(
                bootstrap_values[metric],
                upper_percentile,
            )
        )

    return result

# ============================================================
# 4. RUN ON HELD-OUT TEST
# ============================================================

threshold_ci_result = (
    bootstrap_ci_threshold_metrics_stratified(
        y_true=y_test,
        y_score=test_probability,
        threshold=thr_use,
        n_boot=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
        ci=CONFIDENCE_LEVEL,
    )
)

# ============================================================
# 5. SAVE RESULTS
# ============================================================

threshold_ci_table = pd.DataFrame(
    [threshold_ci_result]
)

threshold_ci_csv = (
    data_path('Final60_Blind_Family_Probe/Final60_heldout_threshold_metrics_bootstrap_CI.csv')
)

threshold_ci_table.to_csv(
    threshold_ci_csv,
    index=False,
)

# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("FINAL60 HELD-OUT TEST: FIXED-THRESHOLD BOOTSTRAP 95% CI")
print("=" * 90)

print(
    f"Fixed threshold τ       : "
    f"{threshold_ci_result['threshold']:.2f}"
)

print(
    f"Held-out samples        : "
    f"{threshold_ci_result['n_samples']} "
    f"(NADES = {threshold_ci_result['n_NADES']}, "
    f"non-NADES = {threshold_ci_result['n_non_NADES']})"
)

for metric, label in [
    ("Precision", "Precision"),
    ("Recall_TPR", "Recall / TPR"),
    ("F1", "F1"),
    ("Specificity_TNR", "Specificity / TNR"),
    ("False_positive_rate", "False positive rate"),
    ("Balanced_accuracy", "Balanced accuracy"),
]:
    print(
        f"{label:<23}: "
        f"{threshold_ci_result[metric]:.3f} "
        f"[{threshold_ci_result[f'{metric}_CI_low']:.3f}, "
        f"{threshold_ci_result[f'{metric}_CI_high']:.3f}]"
    )

print("\nConfusion matrix:")
print(
    np.array(
        [
            [
                threshold_ci_result["TN"],
                threshold_ci_result["FP"],
            ],
            [
                threshold_ci_result["FN"],
                threshold_ci_result["TP"],
            ],
        ]
    )
)

print("\nSaved:")
print(" -", threshold_ci_csv)

