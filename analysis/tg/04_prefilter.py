
"""Train-only supervised descriptor prefiltering for NADES Tg regression.

The legacy target column is named ``activity`` but contains continuous Tg.
No binary-classification metric is used in this script.

Supervised ranking, imputation and correlation pruning are fitted only on the
training partition.  The internal-evaluation partition is excluded before any
calculation that uses Tg values.
"""

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
from typing import Optional
import warnings

import numpy as np
import pandas as pd

from scipy.stats import rankdata, t
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# ============================================================
# 0. CONFIGURATION
# ============================================================

BASE = data_path('TG')

TRAIN_PATH = data_path('TG/master_train_features.csv')

SPLIT_OUT = data_path('TG/TG_clean_split_seed42.csv')
TECHNICAL_FEATURES_OUT = data_path('TG/TG_technical_clean_feature_names.txt')
ALL_SCORES_OUT = data_path('TG/TG_trainonly_prefilter_scores.csv')
GA_CANDIDATES_OUT = data_path('TG/TG_GA_candidate_descriptors_1000.csv')
GA_NAMES_OUT = data_path('TG/TG_GA_candidate_descriptor_names.txt')
SUMMARY_OUT = data_path('TG/TG_prefilter_summary.csv')

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"  # Legacy name; stores continuous Tg.

# All aqueous concentrations of one parent AB/ABC/ABCD NADES share this ID.
# It is metadata for leakage-free splitting, never a model descriptor.
GROUP_COLUMN = "Core_NADES_ID"

SEED = 42
TEST_SIZE = 0.20
N_REGRESSION_STRATA = 5

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

MAX_MISSING_FRACTION = 0.95
PRE_CORRELATION_POOL_SIZE = 4000
FINAL_GA_POOL_SIZE = 1000
CORRELATION_THRESHOLD = 0.95

# A descriptor is eligible when it satisfies at least one criterion below.
MIN_ABS_SPEARMAN = 0.15
MIN_UNIVARIATE_R2 = 0.03
FDR_THRESHOLD = 0.05

WEIGHT_SPEARMAN = 0.35
WEIGHT_MI = 0.30
WEIGHT_R2 = 0.25
WEIGHT_SIGNIFICANCE = 0.10

EPS = 1e-12


# ============================================================
# 1. HELPERS
# ============================================================

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def percentile_score(values: pd.Series) -> pd.Series:
    """Convert a score to percentile ranks in [0, 1]."""
    clean = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return clean.rank(method="average", pct=True)


def assign_feature_family(feature: str) -> str:
    """Assign each descriptor to its encoding family."""
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


def get_base_descriptor(feature: str) -> str:
    """Recover an approximate underlying molecular descriptor name."""
    prefixes = [
        "AB_avg_", "AB_diff_", "AB_prod_", "AB_ratio_",
        "avg_", "mix_", "A_", "B_", "C_", "D_",
    ]
    for prefix in prefixes:
        if feature.startswith(prefix):
            return feature[len(prefix):]
    return feature


def correlation_and_p_values(
    matrix: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Pearson correlations and two-sided p-values."""
    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)

    x_centered = matrix - matrix.mean(axis=0, keepdims=True)
    y_centered = target - target.mean()

    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(
        np.sum(x_centered ** 2, axis=0)
        * np.sum(y_centered ** 2)
    )

    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros(matrix.shape[1], dtype=float),
        where=denominator > EPS,
    )
    correlation = np.clip(correlation, -1.0, 1.0)

    degrees_of_freedom = len(target) - 2
    if degrees_of_freedom <= 0:
        return correlation, np.ones(matrix.shape[1], dtype=float)

    t_statistic = np.divide(
        correlation * np.sqrt(degrees_of_freedom),
        np.sqrt(np.maximum(1.0 - correlation ** 2, EPS)),
    )
    p_values = 2.0 * t.sf(np.abs(t_statistic), df=degrees_of_freedom)
    p_values[~np.isfinite(p_values)] = 1.0
    return correlation, p_values


def make_regression_strata(y: pd.Series) -> Optional[np.ndarray]:
    """Create approximate Tg quantile strata for a row-wise fallback split."""
    requested_bins = min(N_REGRESSION_STRATA, max(2, len(y) // 20))

    try:
        strata = pd.qcut(
            y,
            q=requested_bins,
            duplicates="drop",
        )
        counts = strata.value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            return strata.astype(str).to_numpy()
    except ValueError:
        pass

    return None


# ============================================================
# 2. LOAD AND VALIDATE Tg DATA
# ============================================================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training file not found:\n{TRAIN_PATH}")

train = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

required_columns = {CODE_COLUMN, TARGET_COLUMN}
if GROUP_COLUMN is not None:
    required_columns.add(GROUP_COLUMN)

missing_required = required_columns - set(train.columns)
if missing_required:
    raise ValueError(f"Missing required columns: {sorted(missing_required)}")

train[CODE_COLUMN] = train[CODE_COLUMN].astype("string").str.strip()
train[TARGET_COLUMN] = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")

if train[CODE_COLUMN].isna().any() or train[CODE_COLUMN].duplicated().any():
    raise ValueError("Code must be present and unique.")

if train[TARGET_COLUMN].isna().any():
    raise ValueError("activity contains missing or non-numeric Tg values.")

y_full = train[TARGET_COLUMN].astype(float).reset_index(drop=True)

print("=" * 92)
print("TG REGRESSION: CLEAN-SPLIT, TRAIN-ONLY SUPERVISED PREFILTERING")
print("=" * 92)
print(f"\nRaw labelled dataset shape: {train.shape}")
print(
    f"Tg range: {y_full.min():.4f} to {y_full.max():.4f} | "
    f"mean = {y_full.mean():.4f}"
)


# ============================================================
# 3. FIXED SPLIT BEFORE ANY SUPERVISED OPERATION
# ============================================================

all_indices = np.arange(len(train))

if GROUP_COLUMN is not None:
    groups = train[GROUP_COLUMN].astype("string").str.strip()
    if groups.isna().any():
        raise ValueError(f"{GROUP_COLUMN} contains missing values.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    train_indices, test_indices = next(
        splitter.split(train, y_full, groups=groups)
    )
    split_method = f"GroupShuffleSplit by {GROUP_COLUMN}"

else:
    warnings.warn(
        "GROUP_COLUMN is None: this is a row-wise exploratory split. "
        "For aqueous Tg data, add Core_NADES_ID and rerun with a group split "
        "before reporting final performance.",
        stacklevel=2,
    )

    strata = make_regression_strata(y_full)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=strata,
    )
    split_method = "Quantile-stratified row-wise split (exploratory only)"

split_table = pd.DataFrame(
    {
        "row_index": all_indices,
        CODE_COLUMN: train[CODE_COLUMN].astype(str).to_numpy(),
        TARGET_COLUMN: y_full.to_numpy(),
        "split": np.where(
            np.isin(all_indices, train_indices),
            "Train",
            "Internal_evaluation",
        ),
    }
)

if GROUP_COLUMN is not None:
    split_table[GROUP_COLUMN] = train[GROUP_COLUMN].astype(str).to_numpy()

split_table.to_csv(SPLIT_OUT, index=False)

print("\nFixed clean split:")
print(f"Method = {split_method}")
print(
    f"Train = {len(train_indices)} | "
    f"Internal evaluation = {len(test_indices)} | seed = {SEED}"
)


# ============================================================
# 4. LABEL-FREE TECHNICAL CLEANING
# ============================================================

drop_columns = [CODE_COLUMN, TARGET_COLUMN]
if GROUP_COLUMN is not None:
    drop_columns.append(GROUP_COLUMN)

X_full = train.drop(columns=drop_columns, errors="ignore").copy()
X_full = X_full.apply(pd.to_numeric, errors="coerce")
X_full = X_full.replace([np.inf, -np.inf], np.nan)

initial_feature_count = X_full.shape[1]

all_nan_columns = X_full.columns[X_full.isna().all()].tolist()
X_full = X_full.drop(columns=all_nan_columns)

high_missing_columns = X_full.columns[
    X_full.isna().mean() > MAX_MISSING_FRACTION
].tolist()
X_full = X_full.drop(columns=high_missing_columns)

constant_columns = X_full.columns[
    X_full.nunique(dropna=True) <= 1
].tolist()
X_full = X_full.drop(columns=constant_columns)

print("\nSearching for exact duplicated columns...")
before_duplicates = X_full.shape[1]
duplicate_mask = X_full.T.duplicated(keep="first")
duplicate_columns = X_full.columns[duplicate_mask].tolist()
X_full = X_full.loc[:, ~duplicate_mask]
exact_duplicate_count = before_duplicates - X_full.shape[1]

with open(TECHNICAL_FEATURES_OUT, "w", encoding="utf-8") as file_handle:
    for feature in X_full.columns:
        file_handle.write(feature + "\n")

print("\nTechnical cleaning summary")
print("-" * 92)
print(f"Initial descriptors      : {initial_feature_count}")
print(f"All-NaN removed          : {len(all_nan_columns)}")
print(f">95% missing removed     : {len(high_missing_columns)}")
print(f"Constant removed         : {len(constant_columns)}")
print(f"Exact duplicates removed : {exact_duplicate_count}")
print(f"Technically clean matrix : {X_full.shape}")


# ============================================================
# 5. TRAIN-ONLY IMPUTATION AND REGRESSION RANKING
# ============================================================

X_train = X_full.iloc[train_indices].reset_index(drop=True)
y_train = y_full.iloc[train_indices].reset_index(drop=True)

imputer = SimpleImputer(strategy="median")
X_imputed_array = imputer.fit_transform(X_train)
X_imputed = pd.DataFrame(
    X_imputed_array,
    columns=X_train.columns,
    index=X_train.index,
)

feature_names = np.asarray(X_train.columns)
y_array = y_train.to_numpy(dtype=float)

print(f"\nTraining matrix for supervised prefilter: {X_imputed.shape}")
print("Calculating Pearson/Spearman associations and p-values...")

pearson_r, pearson_p = correlation_and_p_values(
    X_imputed_array,
    y_array,
)

# Spearman is Pearson correlation of ranks.
ranked_X = rankdata(X_imputed_array, axis=0)
ranked_y = rankdata(y_array)
spearman_rho, spearman_p = correlation_and_p_values(ranked_X, ranked_y)

univariate_r2 = pearson_r ** 2
spearman_fdr = benjamini_hochberg(spearman_p)
negative_log10_fdr = -np.log10(np.clip(spearman_fdr, 1e-300, 1.0))

print("Calculating mutual information for regression...")
mutual_information = mutual_info_regression(
    X_imputed_array,
    y_array,
    discrete_features=False,
    random_state=SEED,
    n_neighbors=3,
)


# ============================================================
# 6. SCORE TABLE AND ELIGIBILITY
# ============================================================

scores = pd.DataFrame(
    {
        "feature": feature_names,
        "family": [assign_feature_family(feature) for feature in feature_names],
        "base_descriptor": [get_base_descriptor(feature) for feature in feature_names],
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "abs_spearman_rho": np.abs(spearman_rho),
        "spearman_p": spearman_p,
        "spearman_fdr": spearman_fdr,
        "negative_log10_fdr": negative_log10_fdr,
        "univariate_linear_r2": univariate_r2,
        "mutual_information_regression": mutual_information,
        "direction": np.where(
            spearman_rho > 0,
            "increases_Tg",
            np.where(spearman_rho < 0, "decreases_Tg", "neutral"),
        ),
    }
)

scores["score_spearman"] = percentile_score(scores["abs_spearman_rho"])
scores["score_mutual_information"] = percentile_score(
    scores["mutual_information_regression"]
)
scores["score_univariate_r2"] = percentile_score(scores["univariate_linear_r2"])
scores["score_significance"] = percentile_score(scores["negative_log10_fdr"])

scores["informativeness_index"] = (
    WEIGHT_SPEARMAN * scores["score_spearman"]
    + WEIGHT_MI * scores["score_mutual_information"]
    + WEIGHT_R2 * scores["score_univariate_r2"]
    + WEIGHT_SIGNIFICANCE * scores["score_significance"]
)

scores["passes_spearman_filter"] = (
    scores["abs_spearman_rho"] >= MIN_ABS_SPEARMAN
)
scores["passes_r2_filter"] = (
    scores["univariate_linear_r2"] >= MIN_UNIVARIATE_R2
)
scores["passes_fdr_mi_filter"] = (
    (scores["spearman_fdr"] < FDR_THRESHOLD)
    & (scores["mutual_information_regression"] > 0)
)
scores["eligible_for_prefilter"] = scores[
    [
        "passes_spearman_filter",
        "passes_r2_filter",
        "passes_fdr_mi_filter",
    ]
].any(axis=1)

scores = scores.sort_values(
    by=[
        "informativeness_index",
        "abs_spearman_rho",
        "mutual_information_regression",
    ],
    ascending=False,
).reset_index(drop=True)

scores.to_csv(ALL_SCORES_OUT, index=False)


# ============================================================
# 7. TRAIN-ONLY CORRELATION PRUNING
# ============================================================

eligible_scores = scores[scores["eligible_for_prefilter"]].copy()

if len(eligible_scores) < FINAL_GA_POOL_SIZE:
    warnings.warn(
        f"Only {len(eligible_scores)} descriptors passed the eligibility "
        "filter. The ranked full technical set will be used as fallback.",
        stacklevel=2,
    )
    ranking_source = scores
else:
    ranking_source = eligible_scores

pre_pool_size = min(PRE_CORRELATION_POOL_SIZE, len(ranking_source))
pre_pool_features = ranking_source.head(pre_pool_size)["feature"].tolist()

print(f"\nEligible descriptors: {len(eligible_scores)}")
print(f"Pre-correlation ranked pool: {len(pre_pool_features)}")
print(
    "Calculating correlation matrix for "
    f"{len(pre_pool_features)} descriptors..."
)

correlation_matrix = X_imputed[pre_pool_features].corr(method="pearson").abs()

selected_features: list[str] = []
removed_for_correlation: list[dict] = []

for feature in pre_pool_features:
    if not selected_features:
        selected_features.append(feature)
        continue

    correlations = correlation_matrix.loc[feature, selected_features]
    maximum_correlation = float(correlations.max())

    if maximum_correlation < CORRELATION_THRESHOLD:
        selected_features.append(feature)
    else:
        removed_for_correlation.append(
            {
                "removed_feature": feature,
                "retained_correlated_feature": correlations.idxmax(),
                "absolute_correlation": maximum_correlation,
            }
        )

    if len(selected_features) >= FINAL_GA_POOL_SIZE:
        break

if len(selected_features) < FINAL_GA_POOL_SIZE:
    warnings.warn(
        f"Only {len(selected_features)} descriptors remained after "
        f"correlation pruning; requested {FINAL_GA_POOL_SIZE}.",
        stacklevel=2,
    )


# ============================================================
# 8. SAVE CANDIDATES AND SUMMARY
# ============================================================

candidate_table = scores[scores["feature"].isin(selected_features)].copy()
selection_order = {
    feature: order + 1
    for order, feature in enumerate(selected_features)
}
candidate_table["GA_candidate_order"] = candidate_table["feature"].map(selection_order)
candidate_table = candidate_table.sort_values("GA_candidate_order").reset_index(drop=True)

candidate_table.to_csv(GA_CANDIDATES_OUT, index=False)

with open(GA_NAMES_OUT, "w", encoding="utf-8") as file_handle:
    for feature in selected_features:
        file_handle.write(feature + "\n")

summary_records = [
    {"metric": "target", "value": "Tg stored in activity"},
    {"metric": "split_method", "value": split_method},
    {"metric": "full_labelled_samples", "value": len(train)},
    {"metric": "training_samples_used_for_supervised_prefilter", "value": len(train_indices)},
    {"metric": "internal_evaluation_samples_excluded_from_supervised_prefilter", "value": len(test_indices)},
    {"metric": "split_random_state", "value": SEED},
    {"metric": "initial_descriptors", "value": initial_feature_count},
    {"metric": "clean_descriptors", "value": X_full.shape[1]},
    {"metric": "eligible_descriptors", "value": len(eligible_scores)},
    {"metric": "pre_correlation_pool", "value": len(pre_pool_features)},
    {"metric": "final_GA_candidate_pool", "value": len(selected_features)},
    {"metric": "correlation_threshold", "value": CORRELATION_THRESHOLD},
    {"metric": "median_abs_spearman_final_pool", "value": candidate_table["abs_spearman_rho"].median()},
    {"metric": "median_MI_final_pool", "value": candidate_table["mutual_information_regression"].median()},
    {"metric": "median_univariate_R2_final_pool", "value": candidate_table["univariate_linear_r2"].median()},
    {"metric": "median_informativeness_index", "value": candidate_table["informativeness_index"].median()},
]

pd.DataFrame(summary_records).to_csv(SUMMARY_OUT, index=False)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n" + "=" * 92)
print("TG REGRESSION PREFILTER RESULTS")
print("=" * 92)
print(
    f"\nSupervised prefilter samples: {len(y_array)} "
    f"(internal-evaluation samples excluded: {len(test_indices)})"
)
print(f"Clean descriptor count       : {X_full.shape[1]}")
print(f"Eligible before correlation  : {len(eligible_scores)}")
print(f"Ranked pre-correlation pool  : {len(pre_pool_features)}")
print(f"Final GA candidate pool      : {len(selected_features)}")
print(f"Descriptors removed by correlation: {len(removed_for_correlation)}")

print("\nRepresentation families in final pool:")
print(candidate_table["family"].value_counts())

print("\nTop 30 candidate descriptors:")
print(
    candidate_table[
        [
            "GA_candidate_order",
            "feature",
            "family",
            "direction",
            "spearman_rho",
            "mutual_information_regression",
            "univariate_linear_r2",
            "spearman_fdr",
            "informativeness_index",
        ]
    ].head(30).to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}",
    )
)

print("\nSaved:")
for path in [
    SPLIT_OUT,
    TECHNICAL_FEATURES_OUT,
    ALL_SCORES_OUT,
    GA_CANDIDATES_OUT,
    GA_NAMES_OUT,
    SUMMARY_OUT,
]:
    print(" -", path)

print("\n✅ Tg train-only regression prefiltering completed.")

