
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
# HELD-OUT TEST: STRATIFIED BOOTSTRAP 95% CONFIDENCE INTERVALS
# Final60 XGBoost model
#
# Uses:
#   y_test
#   test_probability
#   OUTPUT_DIR
#
# Saves:
#   Final60_heldout_bootstrap_CI.csv
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# 1. SETTINGS
# ============================================================

N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95
POSITIVE_LABEL = 1  # NADES

# ============================================================
# 2. STRATIFIED BOOTSTRAP FUNCTION
# ============================================================

def bootstrap_ci_auc_stratified(
    y_true,
    y_score,
    n_boot=5000,
    seed=42,
    ci=0.95,
    pos_label=1,
):
    """
    Stratified percentile bootstrap confidence intervals for:
      - ROC-AUC
      - Average precision (PR-AUC) of the positive class.

    Stratification preserves the original number of positive
    and negative samples in every bootstrap resample.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(
        y_score,
        dtype=float,
    )

    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score must have the same length. "
            f"Got {len(y_true)} and {len(y_score)}."
        )

    if len(y_true) == 0:
        raise ValueError(
            "y_true and y_score cannot be empty."
        )

    if pd.isna(y_true).any():
        raise ValueError(
            "y_true contains missing values."
        )

    if not np.isfinite(y_score).all():
        raise ValueError(
            "y_score contains missing or non-finite values."
        )

    if np.any(
        (y_score < 0.0)
        | (y_score > 1.0)
    ):
        raise ValueError(
            "y_score must contain probabilities between 0 and 1."
        )

    unique_labels = np.unique(
        y_true
    )

    if len(unique_labels) != 2:
        raise ValueError(
            "y_true must contain exactly two classes. "
            f"Found: {unique_labels}"
        )

    if pos_label not in unique_labels:
        raise ValueError(
            f"pos_label={pos_label} is not in y_true. "
            f"Found labels: {unique_labels}"
        )

    # Recode: NADES (positive class) = 1.
    y_binary = (
        y_true == pos_label
    ).astype(
        np.int8
    )

    index_positive = np.where(
        y_binary == 1
    )[0]

    index_negative = np.where(
        y_binary == 0
    )[0]

    n_positive = len(
        index_positive
    )

    n_negative = len(
        index_negative
    )

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Both classes must be present."
        )

    # Point estimates on original held-out test set.
    roc_auc_point = roc_auc_score(
        y_binary,
        y_score,
    )

    average_precision_point = average_precision_score(
        y_binary,
        y_score,
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_roc_auc = np.empty(
        n_boot,
        dtype=float,
    )

    bootstrap_average_precision = np.empty(
        n_boot,
        dtype=float,
    )

    for bootstrap_index in range(n_boot):

        sampled_positive = rng.choice(
            index_positive,
            size=n_positive,
            replace=True,
        )

        sampled_negative = rng.choice(
            index_negative,
            size=n_negative,
            replace=True,
        )

        sampled_index = np.concatenate(
            [
                sampled_positive,
                sampled_negative,
            ]
        )

        y_bootstrap = y_binary[
            sampled_index
        ]

        probability_bootstrap = y_score[
            sampled_index
        ]

        bootstrap_roc_auc[
            bootstrap_index
        ] = roc_auc_score(
            y_bootstrap,
            probability_bootstrap,
        )

        bootstrap_average_precision[
            bootstrap_index
        ] = average_precision_score(
            y_bootstrap,
            probability_bootstrap,
        )

    alpha = (
        1.0 - ci
    ) / 2.0

    lower_percentile = 100.0 * alpha
    upper_percentile = 100.0 * (1.0 - alpha)

    return {
        "dataset": "Held-out test",
        "positive_class": "NADES",
        "positive_label": int(pos_label),
        "n_samples": int(
            n_positive + n_negative
        ),
        "n_NADES": int(n_positive),
        "n_non_NADES": int(n_negative),
        "NADES_prevalence": float(
            n_positive
            / (n_positive + n_negative)
        ),
        "n_bootstrap": int(n_boot),
        "confidence_level": float(ci),
        "ROC_AUC": float(roc_auc_point),
        "ROC_AUC_CI_low": float(
            np.percentile(
                bootstrap_roc_auc,
                lower_percentile,
            )
        ),
        "ROC_AUC_CI_high": float(
            np.percentile(
                bootstrap_roc_auc,
                upper_percentile,
            )
        ),
        "Average_Precision_NADES": float(
            average_precision_point
        ),
        "Average_Precision_CI_low": float(
            np.percentile(
                bootstrap_average_precision,
                lower_percentile,
            )
        ),
        "Average_Precision_CI_high": float(
            np.percentile(
                bootstrap_average_precision,
                upper_percentile,
            )
        ),
    }

# ============================================================
# 3. CALCULATE HELD-OUT TEST CI
# ============================================================

bootstrap_result = bootstrap_ci_auc_stratified(
    y_true=y_test,
    y_score=test_probability,
    n_boot=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
    ci=CONFIDENCE_LEVEL,
    pos_label=POSITIVE_LABEL,
)

# ============================================================
# 4. SAVE TABLE
# ============================================================

bootstrap_table = pd.DataFrame(
    [bootstrap_result]
)

bootstrap_csv = (
    data_path('Final60_Blind_Family_Probe/Final60_heldout_bootstrap_CI.csv')
)

bootstrap_table.to_csv(
    bootstrap_csv,
    index=False,
)

# ============================================================
# 5. PRINT RESULTS
# ============================================================

print("\n" + "=" * 85)
print("FINAL60 HELD-OUT TEST: STRATIFIED BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 85)

print(
    f"Held-out samples        : "
    f"{bootstrap_result['n_samples']}"
)

print(
    f"NADES / non-NADES       : "
    f"{bootstrap_result['n_NADES']} / "
    f"{bootstrap_result['n_non_NADES']}"
)

print(
    f"NADES prevalence        : "
    f"{bootstrap_result['NADES_prevalence']:.4f}"
)

print(
    f"Bootstrap resamples     : "
    f"{bootstrap_result['n_bootstrap']}"
)

print(
    f"\nTest ROC-AUC            : "
    f"{bootstrap_result['ROC_AUC']:.3f} "
    f"[{bootstrap_result['ROC_AUC_CI_low']:.3f}, "
    f"{bootstrap_result['ROC_AUC_CI_high']:.3f}]"
)

print(
    f"Test AP for NADES       : "
    f"{bootstrap_result['Average_Precision_NADES']:.3f} "
    f"[{bootstrap_result['Average_Precision_CI_low']:.3f}, "
    f"{bootstrap_result['Average_Precision_CI_high']:.3f}]"
)

print("\nSaved:")
print(" -", bootstrap_csv)


