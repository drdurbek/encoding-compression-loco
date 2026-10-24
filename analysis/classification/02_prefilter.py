
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
# CLEAN-SPLIT, TRAIN-ONLY SUPERVISED DESCRIPTOR PREFILTERING
#
# Purpose:
#   Create the 1,000-descriptor GA candidate pool without using labels
#   from the fixed 20% internal evaluation subset.
#
# Workflow:
#   1. Load all 639 labelled mixtures.
#   2. Create and save the fixed stratified 80:20 split (seed = 42).
#   3. Apply technical, label-free descriptor cleaning to the full matrix.
#   4. Fit imputation and calculate d, MI, AUC, Mann-Whitney/FDR,
#      ranking, and correlation pruning on the training partition only.
#
# The saved split file MUST be reused by the subsequent Final GA script.
#
# Output:
#   GA_candidate_descriptors_1000.csv
#
# Ranking signals:
#   - absolute Cohen's d
#   - mutual information
#   - orientation-free univariate ROC-AUC
#   - Mann-Whitney U significance with BH-FDR correction
#
# Redundancy control:
#   greedy correlation pruning at |r| >= 0.95
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scipy.stats import mannwhitneyu


# ============================================================
# 0. CONFIGURATION
# ============================================================

BASE = data_path('')

TRAIN_PATH = data_path('master_train_features.csv')

SPLIT_OUT = data_path('Final_clean_split_seed42.csv')

TECHNICAL_FEATURES_OUT = (
    data_path('Final_technical_clean_feature_names.txt')
)

ALL_SCORES_OUT = (
    data_path('Final_trainonly_prefilter_scores.csv')
)

GA_CANDIDATES_OUT = (
    data_path('GA_candidate_descriptors_1000.csv')
)

GA_NAMES_OUT = (
    data_path('GA_candidate_descriptor_names.txt')
)

SUMMARY_OUT = (
    data_path('prefilter_summary.csv')
)

NA_VALUES = [
    "na", "NA", "Na",
    "N/A", "n/a",
    "None", "none",
    "-", "",
    "inf", "-inf", "Inf", "+inf",
]

SEED = 42
TEST_SIZE = 0.20

MAX_MISSING_FRACTION = 0.95

# Initial ranked pool subjected to correlation pruning
PRE_CORRELATION_POOL_SIZE = 4000

# Final descriptor pool passed to genetic algorithm
FINAL_GA_POOL_SIZE = 1000

# Redundancy threshold
CORRELATION_THRESHOLD = 0.95

# Minimal effect/significance requirements.
# A descriptor can survive when it satisfies at least one
# meaningful signal criterion.
MIN_ABS_D = 0.20
MIN_UNIVARIATE_AUC = 0.55
FDR_THRESHOLD = 0.05

# Composite score weights
WEIGHT_COHENS_D = 0.35
WEIGHT_MI = 0.30
WEIGHT_AUC = 0.25
WEIGHT_SIGNIFICANCE = 0.10

EPS = 1e-12


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def benjamini_hochberg(
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Benjamini-Hochberg FDR adjustment.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n_tests = len(p_values)

    order = np.argsort(
        p_values
    )

    ranked_p = p_values[
        order
    ]

    adjusted_ranked = (
        ranked_p
        * n_tests
        / np.arange(
            1,
            n_tests + 1,
        )
    )

    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted_ranked = np.clip(
        adjusted_ranked,
        0.0,
        1.0,
    )

    adjusted = np.empty(
        n_tests,
        dtype=float,
    )

    adjusted[
        order
    ] = adjusted_ranked

    return adjusted


def percentile_score(
    values: pd.Series,
) -> pd.Series:
    """
    Convert metric values to percentile ranks in [0, 1].
    Larger values receive larger scores.
    """

    values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return values.rank(
        method="average",
        pct=True,
    )


def assign_feature_family(
    feature: str,
) -> str:
    """
    Assign descriptor to representation family.
    """

    if feature.startswith("AB_"):
        return "AB_interactions"

    if feature.startswith("avg_"):
        return "Arithmetic_mean"

    if feature.startswith("mix_"):
        return "Weighted_mixture"

    if feature.startswith("A_"):
        return "Component_A"

    if feature.startswith("B_"):
        return "Component_B"

    if feature.startswith("C_"):
        return "Component_C"

    if feature.startswith("D_"):
        return "Component_D"

    if (
        feature.startswith("has_")
        or feature == "n_components"
        or feature == "val_total"
        or feature.endswith("_val")
        or feature.endswith("_val_filled")
        or feature.endswith("_val_frac")
    ):
        return "Stoichiometry"

    return "Other"


def get_base_descriptor(
    feature: str,
) -> str:
    """
    Recover an approximate underlying descriptor base.
    Useful for examining representation redundancy.
    """

    prefixes = [
        "AB_avg_",
        "AB_diff_",
        "AB_prod_",
        "AB_ratio_",
        "avg_",
        "mix_",
        "A_",
        "B_",
        "C_",
        "D_",
    ]

    for prefix in prefixes:
        if feature.startswith(prefix):
            return feature[
                len(prefix):
            ]

    return feature


# ============================================================
# 2. LOAD TRAINING DATA ONLY
# ============================================================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_PATH}"
    )

train = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

print("=" * 92)
print("CLEAN-SPLIT, TRAIN-ONLY SUPERVISED DESCRIPTOR PREFILTERING")
print("=" * 92)

print("\nRaw labelled dataset shape:", train.shape)


# ============================================================
# 3. VALIDATE TARGET
# ============================================================

required_columns = {
    "Code",
    "activity",
}

missing_required = (
    required_columns
    - set(train.columns)
)

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )

y = pd.to_numeric(
    train["activity"],
    errors="coerce",
)

if y.isna().any():
    raise ValueError(
        "activity contains missing or non-numeric values."
    )

unexpected_classes = (
    set(y.unique())
    - {0, 1}
)

if unexpected_classes:
    raise ValueError(
        "activity must contain only 0 and 1. "
        f"Found: {unexpected_classes}"
    )

y = y.astype(np.int8)

print("\nFull labelled-data class distribution:")
print(
    y.value_counts()
    .sort_index()
)

print("\n0 = non-NADES")
print("1 = NADES")


# ============================================================
# 4. FIXED SPLIT CREATED BEFORE ALL SUPERVISED OPERATIONS
# ============================================================

all_indices = np.arange(len(train))

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    shuffle=True,
    stratify=y.to_numpy(),
)

split_table = pd.DataFrame({
    "row_index": all_indices,
    "Code": train["Code"].astype(str).to_numpy(),
    "activity": y.to_numpy(),
    "split": np.where(
        np.isin(all_indices, train_indices),
        "Train",
        "Internal_evaluation",
    ),
})

split_table.to_csv(SPLIT_OUT, index=False)

print("\nFixed clean split:")
print(
    f"Train = {len(train_indices)} | "
    f"Internal evaluation = {len(test_indices)} | "
    f"seed = {SEED}"
)
print("Train class distribution:")
print(y.iloc[train_indices].value_counts().sort_index())
print("Internal-evaluation class distribution:")
print(y.iloc[test_indices].value_counts().sort_index())


# ============================================================
# 5. FULL FEATURE MATRIX FOR LABEL-FREE TECHNICAL CLEANING
# ============================================================

X = train.drop(
    columns=[
        "Code",
        "activity",
    ],
    errors="ignore",
).copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce",
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

initial_feature_count = X.shape[1]


# ============================================================
# 6. LABEL-FREE TECHNICAL CLEANING
# ============================================================

# 5.1 All-NaN
all_nan_columns = X.columns[
    X.isna().all()
].tolist()

X = X.drop(
    columns=all_nan_columns
)


# 5.2 Excessive missingness
high_missing_columns = X.columns[
    X.isna().mean()
    > MAX_MISSING_FRACTION
].tolist()

X = X.drop(
    columns=high_missing_columns
)


# 5.3 Constant columns
constant_columns = X.columns[
    X.nunique(
        dropna=True
    ) <= 1
].tolist()

X = X.drop(
    columns=constant_columns
)


# 5.4 Exact duplicates
print("\nSearching for exact duplicated columns...")

before_duplicates = X.shape[1]

duplicate_mask = X.T.duplicated(
    keep="first"
)

duplicate_columns = X.columns[
    duplicate_mask
].tolist()

X = X.loc[
    :,
    ~duplicate_mask
]

exact_duplicate_count = (
    before_duplicates
    - X.shape[1]
)

print("\nCleaning summary")
print("-" * 92)
print(
    f"Initial descriptors     : "
    f"{initial_feature_count}"
)
print(
    f"All-NaN removed         : "
    f"{len(all_nan_columns)}"
)
print(
    f">95% missing removed    : "
    f"{len(high_missing_columns)}"
)
print(
    f"Constant removed        : "
    f"{len(constant_columns)}"
)
print(
    f"Exact duplicates removed: "
    f"{exact_duplicate_count}"
)
print(
    f"Technically clean full matrix: "
    f"{X.shape}"
)


# This feature manifest is label-free and must be reused downstream so that
# train and internal-evaluation matrices have identical technical columns.
with open(
    TECHNICAL_FEATURES_OUT,
    "w",
    encoding="utf-8",
) as file_handle:
    for feature in X.columns:
        file_handle.write(feature + "\n")

# From this line onward, every calculation that uses y is performed on the
# training partition only. The internal-evaluation subset is not accessed.
X = X.iloc[train_indices].reset_index(drop=True)
y = y.iloc[train_indices].reset_index(drop=True)

print(
    f"Training matrix for supervised prefilter: {X.shape}"
)


# ============================================================
# 7. TRAIN-ONLY MEDIAN IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_imputed_array = imputer.fit_transform(
    X
)

X_imputed = pd.DataFrame(
    X_imputed_array,
    columns=X.columns,
    index=X.index,
)

feature_names = np.asarray(
    X.columns,
)

y_array = y.to_numpy()

nades_mask = (
    y_array == 1
)

non_nades_mask = (
    y_array == 0
)

X_nades = X_imputed_array[
    nades_mask
]

X_non_nades = X_imputed_array[
    non_nades_mask
]

n_nades = X_nades.shape[0]
n_non_nades = X_non_nades.shape[0]


# ============================================================
# 7. COHEN'S d
# ============================================================

print("\nCalculating Cohen's d...")

mean_nades = np.mean(
    X_nades,
    axis=0,
)

mean_non_nades = np.mean(
    X_non_nades,
    axis=0,
)

variance_nades = np.var(
    X_nades,
    axis=0,
    ddof=1,
)

variance_non_nades = np.var(
    X_non_nades,
    axis=0,
    ddof=1,
)

pooled_variance = (
    (
        (n_nades - 1)
        * variance_nades
    )
    +
    (
        (n_non_nades - 1)
        * variance_non_nades
    )
) / (
    n_nades
    + n_non_nades
    - 2
)

pooled_sd = np.sqrt(
    pooled_variance
)

cohens_d = np.divide(
    mean_nades
    - mean_non_nades,
    pooled_sd,
    out=np.zeros_like(
        mean_nades,
        dtype=float,
    ),
    where=pooled_sd > EPS,
)

absolute_cohens_d = np.abs(
    cohens_d
)


# ============================================================
# 8. MUTUAL INFORMATION
# ============================================================

print("Calculating mutual information...")

mutual_information = mutual_info_classif(
    X_imputed_array,
    y_array,
    discrete_features=False,
    random_state=SEED,
    n_neighbors=3,
)


# ============================================================
# 9. UNIVARIATE ROC-AUC
#
# An orientation-free score is used:
#
# max(AUC, 1 - AUC)
#
# Thus:
#   0.50 = no discrimination
#   1.00 = perfect discrimination
# ============================================================

print("Calculating univariate ROC-AUC...")

raw_auc_values = np.full(
    X_imputed_array.shape[1],
    np.nan,
    dtype=float,
)

oriented_auc_values = np.full(
    X_imputed_array.shape[1],
    np.nan,
    dtype=float,
)

for feature_index in range(
    X_imputed_array.shape[1]
):
    values = X_imputed_array[
        :,
        feature_index,
    ]

    try:
        raw_auc = roc_auc_score(
            y_array,
            values,
        )

        raw_auc_values[
            feature_index
        ] = raw_auc

        oriented_auc_values[
            feature_index
        ] = max(
            raw_auc,
            1.0 - raw_auc,
        )

    except ValueError:
        raw_auc_values[
            feature_index
        ] = 0.5

        oriented_auc_values[
            feature_index
        ] = 0.5


# ============================================================
# 10. MANN-WHITNEY U TEST
# ============================================================

print("Calculating Mann-Whitney U tests...")

mann_whitney_p = np.ones(
    X_imputed_array.shape[1],
    dtype=float,
)

mann_whitney_u = np.full(
    X_imputed_array.shape[1],
    np.nan,
    dtype=float,
)

for feature_index in range(
    X_imputed_array.shape[1]
):
    positive_values = X_nades[
        :,
        feature_index,
    ]

    negative_values = X_non_nades[
        :,
        feature_index,
    ]

    try:
        test_result = mannwhitneyu(
            positive_values,
            negative_values,
            alternative="two-sided",
            method="auto",
        )

        mann_whitney_u[
            feature_index
        ] = test_result.statistic

        mann_whitney_p[
            feature_index
        ] = test_result.pvalue

    except ValueError:
        mann_whitney_u[
            feature_index
        ] = np.nan

        mann_whitney_p[
            feature_index
        ] = 1.0


mann_whitney_fdr = benjamini_hochberg(
    mann_whitney_p
)

negative_log10_fdr = (
    -np.log10(
        np.clip(
            mann_whitney_fdr,
            1e-300,
            1.0,
        )
    )
)


# ============================================================
# 11. CREATE FULL SCORE TABLE
# ============================================================

scores = pd.DataFrame(
    {
        "feature": feature_names,

        "family": [
            assign_feature_family(
                feature
            )
            for feature in feature_names
        ],

        "base_descriptor": [
            get_base_descriptor(
                feature
            )
            for feature in feature_names
        ],

        "mean_NADES": mean_nades,
        "mean_non_NADES": mean_non_nades,

        "cohens_d": cohens_d,
        "abs_cohens_d": absolute_cohens_d,

        "orientation": np.where(
            cohens_d > 0,
            "NADES_oriented",
            np.where(
                cohens_d < 0,
                "non_NADES_oriented",
                "neutral",
            ),
        ),

        "mutual_information": (
            mutual_information
        ),

        "raw_univariate_auc": (
            raw_auc_values
        ),

        "orientation_free_auc": (
            oriented_auc_values
        ),

        "mann_whitney_u": (
            mann_whitney_u
        ),

        "mann_whitney_p": (
            mann_whitney_p
        ),

        "mann_whitney_fdr": (
            mann_whitney_fdr
        ),

        "negative_log10_fdr": (
            negative_log10_fdr
        ),
    }
)


# ============================================================
# 12. NORMALIZED COMPONENT SCORES
# ============================================================

scores[
    "score_cohens_d"
] = percentile_score(
    scores["abs_cohens_d"]
)

scores[
    "score_mutual_information"
] = percentile_score(
    scores["mutual_information"]
)

# Use excess discrimination above chance
scores[
    "auc_above_chance"
] = (
    scores[
        "orientation_free_auc"
    ]
    - 0.5
).clip(
    lower=0.0
)

scores[
    "score_univariate_auc"
] = percentile_score(
    scores["auc_above_chance"]
)

scores[
    "score_significance"
] = percentile_score(
    scores["negative_log10_fdr"]
)


# ============================================================
# 13. COMPOSITE INFORMATIVENESS INDEX
# ============================================================

scores[
    "informativeness_index"
] = (
    WEIGHT_COHENS_D
    * scores[
        "score_cohens_d"
    ]
    +
    WEIGHT_MI
    * scores[
        "score_mutual_information"
    ]
    +
    WEIGHT_AUC
    * scores[
        "score_univariate_auc"
    ]
    +
    WEIGHT_SIGNIFICANCE
    * scores[
        "score_significance"
    ]
)


# ============================================================
# 14. BASIC ELIGIBILITY FILTER
#
# Keep a descriptor when at least one of the following is true:
#   - |d| >= 0.20
#   - orientation-free AUC >= 0.55
#   - FDR < 0.05 and MI > 0
# ============================================================

scores[
    "passes_effect_filter"
] = (
    scores["abs_cohens_d"]
    >= MIN_ABS_D
)

scores[
    "passes_auc_filter"
] = (
    scores["orientation_free_auc"]
    >= MIN_UNIVARIATE_AUC
)

scores[
    "passes_fdr_mi_filter"
] = (
    (
        scores[
            "mann_whitney_fdr"
        ]
        < FDR_THRESHOLD
    )
    &
    (
        scores[
            "mutual_information"
        ] > 0
    )
)

scores[
    "eligible_for_prefilter"
] = (
    scores[
        [
            "passes_effect_filter",
            "passes_auc_filter",
            "passes_fdr_mi_filter",
        ]
    ]
    .any(axis=1)
)

scores = scores.sort_values(
    by=[
        "informativeness_index",
        "abs_cohens_d",
        "mutual_information",
    ],
    ascending=False,
).reset_index(drop=True)

scores.to_csv(
    ALL_SCORES_OUT,
    index=False,
)


# ============================================================
# 15. SELECT PRE-CORRELATION POOL
# ============================================================

eligible_scores = scores[
    scores[
        "eligible_for_prefilter"
    ]
].copy()

if len(eligible_scores) == 0:
    raise RuntimeError(
        "No descriptors passed the basic prefilter."
    )

pre_pool_size = min(
    PRE_CORRELATION_POOL_SIZE,
    len(eligible_scores),
)

pre_pool = eligible_scores.head(
    pre_pool_size
).copy()

pre_pool_features = (
    pre_pool["feature"]
    .tolist()
)

print("\nEligible descriptors:", len(eligible_scores))
print(
    "Pre-correlation ranked pool:",
    len(pre_pool_features),
)


# ============================================================
# 16. GREEDY CORRELATION PRUNING
#
# Features are considered in descending informativeness order.
# A new feature is retained only when its absolute correlation
# with every previously retained feature is below the threshold.
#
# Correlation is calculated only for the ranked pre-pool.
# ============================================================

print(
    "\nCalculating correlation matrix for "
    f"{len(pre_pool_features)} descriptors..."
)

pre_pool_matrix = X_imputed[
    pre_pool_features
]

correlation_matrix = (
    pre_pool_matrix
    .corr(method="pearson")
    .abs()
)

selected_features = []
removed_for_correlation = []

for feature in pre_pool_features:

    if len(selected_features) == 0:
        selected_features.append(
            feature
        )
        continue

    correlations = correlation_matrix.loc[
        feature,
        selected_features,
    ]

    maximum_correlation = float(
        correlations.max()
    )

    if maximum_correlation < CORRELATION_THRESHOLD:
        selected_features.append(
            feature
        )

    else:
        most_correlated_feature = (
            correlations.idxmax()
        )

        removed_for_correlation.append(
            {
                "removed_feature": feature,
                "retained_correlated_feature": (
                    most_correlated_feature
                ),
                "absolute_correlation": (
                    maximum_correlation
                ),
            }
        )

    if len(selected_features) >= FINAL_GA_POOL_SIZE:
        break


# ============================================================
# 17. FALLBACK IF CORRELATION PRUNING RETURNS < TARGET
# ============================================================

if len(selected_features) < FINAL_GA_POOL_SIZE:

    warnings.warn(
        f"Only {len(selected_features)} descriptors remained "
        f"after correlation pruning; requested "
        f"{FINAL_GA_POOL_SIZE}.",
        stacklevel=2,
    )


# ============================================================
# 18. FINAL GA CANDIDATE TABLE
# ============================================================

candidate_table = (
    scores[
        scores["feature"].isin(
            selected_features
        )
    ]
    .copy()
)

selection_order = {
    feature: index + 1
    for index, feature
    in enumerate(
        selected_features
    )
}

candidate_table[
    "GA_candidate_order"
] = candidate_table[
    "feature"
].map(
    selection_order
)

candidate_table = candidate_table.sort_values(
    by="GA_candidate_order"
).reset_index(drop=True)

candidate_table.to_csv(
    GA_CANDIDATES_OUT,
    index=False,
)


# Save plain-text names for convenient later loading
with open(
    GA_NAMES_OUT,
    "w",
    encoding="utf-8",
) as file_handle:
    for feature in selected_features:
        file_handle.write(
            feature + "\n"
        )


# ============================================================
# 19. SUMMARY
# ============================================================

orientation_counts = (
    candidate_table[
        "orientation"
    ]
    .value_counts()
)

family_counts = (
    candidate_table[
        "family"
    ]
    .value_counts()
)

summary_records = [
    {
        "metric": "full_labelled_samples",
        "value": len(train),
    },
    {
        "metric": "training_samples_used_for_supervised_prefilter",
        "value": len(train_indices),
    },
    {
        "metric": "internal_evaluation_samples_excluded_from_supervised_prefilter",
        "value": len(test_indices),
    },
    {
        "metric": "split_random_state",
        "value": SEED,
    },
    {
        "metric": "initial_descriptors",
        "value": initial_feature_count,
    },
    {
        "metric": "clean_descriptors",
        "value": X.shape[1],
    },
    {
        "metric": "eligible_descriptors",
        "value": len(eligible_scores),
    },
    {
        "metric": "pre_correlation_pool",
        "value": len(pre_pool_features),
    },
    {
        "metric": "final_GA_candidate_pool",
        "value": len(selected_features),
    },
    {
        "metric": "correlation_threshold",
        "value": CORRELATION_THRESHOLD,
    },
    {
        "metric": "median_abs_d_final_pool",
        "value": candidate_table[
            "abs_cohens_d"
        ].median(),
    },
    {
        "metric": "median_MI_final_pool",
        "value": candidate_table[
            "mutual_information"
        ].median(),
    },
    {
        "metric": "median_univariate_AUC_final_pool",
        "value": candidate_table[
            "orientation_free_auc"
        ].median(),
    },
    {
        "metric": "median_informativeness_index",
        "value": candidate_table[
            "informativeness_index"
        ].median(),
    },
]

summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    SUMMARY_OUT,
    index=False,
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\n" + "=" * 92)
print("PREFILTER RESULTS")
print("=" * 92)

print(
    f"\nSupervised prefilter samples: {len(y_array)} "
    f"(internal-evaluation samples excluded: {len(test_indices)})"
)

print(
    f"Clean descriptor count       : "
    f"{X.shape[1]}"
)

print(
    f"Eligible before correlation  : "
    f"{len(eligible_scores)}"
)

print(
    f"Ranked pre-correlation pool  : "
    f"{len(pre_pool_features)}"
)

print(
    f"Final GA candidate pool      : "
    f"{len(selected_features)}"
)

print(
    f"Descriptors removed because "
    f"of correlation: "
    f"{len(removed_for_correlation)}"
)

print("\nOrientation in final pool:")
print(
    orientation_counts
)

print("\nRepresentation families in final pool:")
print(
    family_counts
)

print("\nFinal-pool metric summary:")

print(
    candidate_table[
        [
            "abs_cohens_d",
            "mutual_information",
            "orientation_free_auc",
            "mann_whitney_fdr",
            "informativeness_index",
        ]
    ].describe()
)

print("\nTop 30 candidate descriptors:")

print(
    candidate_table[
        [
            "GA_candidate_order",
            "feature",
            "family",
            "orientation",
            "cohens_d",
            "mutual_information",
            "orientation_free_auc",
            "mann_whitney_fdr",
            "informativeness_index",
        ]
    ].head(30).to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}",
    )
)

print("\nSaved:")
print(" -", ALL_SCORES_OUT)
print(" -", GA_CANDIDATES_OUT)
print(" -", GA_NAMES_OUT)
print(" -", SUMMARY_OUT)

print(
    "\n✅ Train-only descriptor prefiltering completed."
)


