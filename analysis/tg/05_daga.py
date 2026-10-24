
"""Diversity-aware GA optimisation for leakage-free NADES Tg regression.

This script uses only the fixed outer-training partition from
TG_clean_split_seed42.csv.  Candidate descriptors were already selected using
only this outer-training data.  GA fitness is calculated with inner GroupKFold
over Core_NADES_ID, so all concentration variants of a parent NADES remain in
the same inner train or validation fold.

The 55-row outer internal-evaluation partition is never used here.

This is the model-selection stage. Its inner OOF scores rank descriptor
subsets *within the outer-training data*; they are not the final reported
generalisation result. After a single subset size is fixed, fit that frozen
RF model on all 208 outer-training rows and evaluate it once on the 55
unseen rows in a separate script.
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


from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


# ============================================================
# 0. PATHS
# ============================================================

BASE = data_path('TG')

TRAIN_PATH = data_path('TG/master_train_features.csv')
CANDIDATE_PATH = data_path('TG/TG_GA_candidate_descriptors_1000.csv')
SPLIT_IN = data_path('TG/TG_clean_split_seed42.csv')

OUTPUT_ROOT = data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SPLIT_OUT = data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_fixed_group_train_test_split.csv')
OVERALL_COMPARISON_OUT = data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_subset_size_comparison.csv')
OVERALL_BEST_FEATURES_OUT = data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_all_subset_best_descriptors.csv')
ALL_RUNS_OUT = data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/TG_RF_all_GA_runs.csv')


# ============================================================
# 1. SETTINGS
# ============================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"  # Legacy name; contains continuous Tg.
GROUP_COLUMN = "Core_NADES_ID"

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

SEED = 42
MODEL_SEED = 82
N_INNER_SPLITS = 5
SUBSET_SIZES = [10, 20, 30, 40, 50, 60]

# The RF is deliberately the first learner because it is the published Tg
# baseline. This first sweep therefore isolates the encoding effect; XGBoost
# can be compared later on the same frozen descriptor set. The search forest
# is modest to keep the GA tractable. Do not tune it on outer evaluation data.
RF_GA_PARAMS = {
    "n_estimators": 180,
    "max_depth": 6,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "random_state": 42,
    "n_jobs": -1,
}

# Optional dry run: set N_GA_RUNS=1, POPULATION_SIZE=10, N_GENERATIONS=3.
# Restore the values below for the reported experiment.
POPULATION_SIZE = 15
N_GENERATIONS = 8
N_GA_RUNS = 5
ELITE_COUNT = 3
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.80
MUTATION_PROBABILITY = 0.35
MIN_MUTATION_REPLACEMENTS = 1
MAX_MUTATION_REPLACEMENTS = 3
EARLY_STOPPING_GENERATIONS = 5
FITNESS_TOLERANCE = 1e-5

# Fitness = OOF R2 + diversity reward - family-concentration penalty.
FITNESS_WEIGHT_R2 = 1.00
FITNESS_WEIGHT_DIVERSITY = 0.10
FAMILY_CONCENTRATION_PENALTY_WEIGHT = 0.05
MAX_DESIRED_FAMILY_FRACTION = 0.45
EPS = 1e-12


# ============================================================
# 2. DATA STRUCTURES AND HELPERS
# ============================================================

Chromosome = tuple


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    oof_r2: float
    oof_rmse: float
    oof_mae: float
    mean_fold_r2: float
    sd_fold_r2: float
    diversity_score: float
    concentration_penalty: float
    family_coverage: float
    family_entropy: float
    maximum_family_fraction: float
    n_families_present: int


def normalise_code(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def build_rf_model():
    return RandomForestRegressor(**RF_GA_PARAMS)


def calculate_diversity(chromosome, candidate_families, all_families):
    selected_families = candidate_families[list(chromosome)]
    counts = Counter(selected_families)
    n_selected = len(chromosome)
    proportions = np.asarray(
        [count / n_selected for count in counts.values()],
        dtype=float,
    )

    family_coverage = len(counts) / max(len(all_families), 1)

    if len(counts) > 1:
        entropy = -np.sum(proportions * np.log(proportions + EPS))
        family_entropy = entropy / np.log(len(counts))
    else:
        family_entropy = 0.0

    maximum_family_fraction = float(proportions.max())
    concentration_penalty = max(
        0.0,
        (maximum_family_fraction - MAX_DESIRED_FAMILY_FRACTION)
        / max(1.0 - MAX_DESIRED_FAMILY_FRACTION, EPS),
    )

    diversity_score = 0.5 * family_coverage + 0.5 * family_entropy

    return {
        "diversity_score": float(diversity_score),
        "concentration_penalty": float(concentration_penalty),
        "family_coverage": float(family_coverage),
        "family_entropy": float(family_entropy),
        "maximum_family_fraction": maximum_family_fraction,
        "n_families_present": int(len(counts)),
    }


def evaluate_chromosome(
    chromosome,
    X_values,
    y_values,
    groups,
    inner_splits,
    candidate_families,
    all_families,
    cache,
):
    """Evaluate one subset using inner group-OOF predictions only.

    The imputer and RF are fitted separately in every GroupKFold training
    partition, so concentration variants of a validation parent NADES are
    never used while fitting that fold.
    """
    if chromosome in cache:
        return cache[chromosome]

    columns = list(chromosome)
    X_subset = X_values[:, columns]
    oof_prediction = np.full(len(y_values), np.nan, dtype=float)
    fold_r2 = []

    for fold_number, (fit_idx, val_idx) in enumerate(inner_splits):
        imputer = SimpleImputer(strategy="median")
        X_fit = imputer.fit_transform(X_subset[fit_idx])
        X_val = imputer.transform(X_subset[val_idx])

        model = build_rf_model()
        model.fit(X_fit, y_values[fit_idx])
        prediction = model.predict(X_val)

        oof_prediction[val_idx] = prediction
        fold_r2.append(r2_score(y_values[val_idx], prediction))

    if np.isnan(oof_prediction).any():
        raise RuntimeError("Inner OOF predictions are incomplete.")

    oof_r2 = r2_score(y_values, oof_prediction)
    oof_rmse = root_mean_squared_error(y_values, oof_prediction)
    oof_mae = mean_absolute_error(y_values, oof_prediction)

    diversity = calculate_diversity(
        chromosome,
        candidate_families,
        all_families,
    )

    fitness = (
        FITNESS_WEIGHT_R2 * oof_r2
        + FITNESS_WEIGHT_DIVERSITY * diversity["diversity_score"]
        - FAMILY_CONCENTRATION_PENALTY_WEIGHT
        * diversity["concentration_penalty"]
    )

    result = FitnessResult(
        fitness=float(fitness),
        oof_r2=float(oof_r2),
        oof_rmse=float(oof_rmse),
        oof_mae=float(oof_mae),
        mean_fold_r2=float(np.mean(fold_r2)),
        sd_fold_r2=float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
        diversity_score=diversity["diversity_score"],
        concentration_penalty=diversity["concentration_penalty"],
        family_coverage=diversity["family_coverage"],
        family_entropy=diversity["family_entropy"],
        maximum_family_fraction=diversity["maximum_family_fraction"],
        n_families_present=diversity["n_families_present"],
    )

    cache[chromosome] = result
    return result


def initialise_population(pool_size, subset_size, rng):
    population = set()
    while len(population) < POPULATION_SIZE:
        chromosome = tuple(sorted(rng.sample(range(pool_size), subset_size)))
        population.add(chromosome)
    return list(population)


def tournament_select(population, results, rng):
    contenders = rng.sample(population, min(TOURNAMENT_SIZE, len(population)))
    return max(
        contenders,
        key=lambda chromosome: (
            results[chromosome].fitness,
            results[chromosome].oof_r2,
        ),
    )


def crossover(parent_a, parent_b, subset_size, pool_size, rng):
    common = set(parent_a).intersection(parent_b)
    union = list(set(parent_a).union(parent_b))
    child = set(common)

    available = [feature for feature in union if feature not in child]
    rng.shuffle(available)

    for feature in available:
        if len(child) >= subset_size:
            break
        child.add(feature)

    if len(child) < subset_size:
        outside = [feature for feature in range(pool_size) if feature not in child]
        child.update(rng.sample(outside, subset_size - len(child)))

    return tuple(sorted(child))


def mutate(chromosome, pool_size, rng):
    subset_size = len(chromosome)
    child = set(chromosome)
    n_replacements = rng.randint(
        MIN_MUTATION_REPLACEMENTS,
        min(MAX_MUTATION_REPLACEMENTS, subset_size),
    )

    removed = rng.sample(list(child), n_replacements)
    child.difference_update(removed)

    replacements = rng.sample(
        [feature for feature in range(pool_size) if feature not in child],
        n_replacements,
    )
    child.update(replacements)

    return tuple(sorted(child))


def run_ga(
    subset_size,
    run_number,
    X_values,
    y_values,
    groups,
    inner_splits,
    candidate_families,
    all_families,
):
    """Run one reproducible GA search for one descriptor dimensionality."""
    rng = random.Random(SEED + 1000 * subset_size + run_number)
    pool_size = X_values.shape[1]
    cache = {}
    population = initialise_population(pool_size, subset_size, rng)

    best_chromosome = None
    best_result = None
    no_improvement = 0
    history = []

    for generation in range(1, N_GENERATIONS + 1):
        results = {}
        for chromosome in population:
            results[chromosome] = evaluate_chromosome(
                chromosome,
                X_values,
                y_values,
                groups,
                inner_splits,
                candidate_families,
                all_families,
                cache,
            )

        ranked_population = sorted(
            population,
            key=lambda chromosome: (
                results[chromosome].fitness,
                results[chromosome].oof_r2,
            ),
            reverse=True,
        )

        generation_best = ranked_population[0]
        generation_result = results[generation_best]

        if (
            best_result is None
            or generation_result.fitness
            > best_result.fitness + FITNESS_TOLERANCE
        ):
            best_chromosome = generation_best
            best_result = generation_result
            no_improvement = 0
        else:
            no_improvement += 1

        history.append(
            {
                "subset_size": subset_size,
                "run": run_number,
                "generation": generation,
                "generation_best_fitness": generation_result.fitness,
                "generation_best_oof_r2": generation_result.oof_r2,
                "best_fitness_so_far": best_result.fitness,
                "best_oof_r2_so_far": best_result.oof_r2,
            }
        )

        print(
            f"Generation {generation:02d} | "
            f"fitness={generation_result.fitness:.4f} | "
            f"OOF R2={generation_result.oof_r2:.4f} | "
            f"RMSE={generation_result.oof_rmse:.3f} | "
            f"no improvement={no_improvement}"
        )

        if no_improvement >= EARLY_STOPPING_GENERATIONS:
            break

        next_population = ranked_population[:ELITE_COUNT]

        while len(next_population) < POPULATION_SIZE:
            parent_a = tournament_select(ranked_population, results, rng)
            parent_b = tournament_select(ranked_population, results, rng)

            if rng.random() < CROSSOVER_PROBABILITY:
                child = crossover(
                    parent_a,
                    parent_b,
                    subset_size,
                    pool_size,
                    rng,
                )
            else:
                child = parent_a

            if rng.random() < MUTATION_PROBABILITY:
                child = mutate(child, pool_size, rng)

            if child not in next_population:
                next_population.append(child)

        population = next_population

    return best_chromosome, best_result, history


# ============================================================
# 3. LOAD FIXED OUTER-TRAINING DATA
# ============================================================

for required_path in [TRAIN_PATH, CANDIDATE_PATH, SPLIT_IN]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found:\n{required_path}")

data = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)
candidate_table = pd.read_csv(CANDIDATE_PATH, low_memory=False)
split_ref = pd.read_csv(SPLIT_IN, low_memory=False)

required_data = {CODE_COLUMN, TARGET_COLUMN, GROUP_COLUMN}
if not required_data.issubset(data.columns):
    raise ValueError(
        f"Dataset must contain {sorted(required_data)}."
    )

required_candidates = {"feature", "family"}
if not required_candidates.issubset(candidate_table.columns):
    raise ValueError(
        f"Candidate table must contain {sorted(required_candidates)}."
    )

if not {CODE_COLUMN, "split"}.issubset(split_ref.columns):
    raise ValueError("Split file must contain Code and split columns.")

data[CODE_COLUMN] = normalise_code(data[CODE_COLUMN])
split_ref[CODE_COLUMN] = normalise_code(split_ref[CODE_COLUMN])

data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
if data[TARGET_COLUMN].isna().any():
    raise ValueError("activity contains missing or non-numeric Tg values.")

if data[CODE_COLUMN].isna().any() or data[CODE_COLUMN].duplicated().any():
    raise ValueError("Code must be non-missing and unique.")

if data[GROUP_COLUMN].isna().any():
    raise ValueError("Core_NADES_ID contains missing values.")

candidate_table = candidate_table.dropna(subset=["feature"]).copy()
candidate_table["feature"] = candidate_table["feature"].astype(str).str.strip()
candidate_table = candidate_table.drop_duplicates("feature", keep="first").reset_index(drop=True)
candidate_features = candidate_table["feature"].tolist()

if len(candidate_features) < max(SUBSET_SIZES):
    raise ValueError("Candidate pool is smaller than the largest subset size.")

missing_features = [
    feature for feature in candidate_features
    if feature not in data.columns
]
if missing_features:
    raise ValueError(
        "Candidate descriptors absent from the dataset: "
        f"{missing_features[:20]}"
    )

split_map = dict(zip(split_ref[CODE_COLUMN], split_ref["split"]))
split_label = data[CODE_COLUMN].map(split_map)
if split_label.isna().any():
    missing_codes = data.loc[split_label.isna(), CODE_COLUMN].head(20).tolist()
    raise ValueError(
        f"{int(split_label.isna().sum())} codes are absent from split file. "
        f"Examples: {missing_codes}"
    )

split_label_normalised = split_label.astype("string").str.strip().str.lower()
train_labels = {"train", "outer_train", "training"}
is_train = split_label_normalised.isin(train_labels).to_numpy()
train_indices = np.where(is_train)[0]
test_indices = np.where(~is_train)[0]

if len(train_indices) == 0 or len(test_indices) == 0:
    raise ValueError("Fixed split has an empty partition.")

known_test_labels = {
    "internal_evaluation", "internal evaluation", "test", "outer_test",
    "heldout", "held_out",
}
unknown_split_labels = sorted(
    set(split_label_normalised.dropna().unique())
    - train_labels
    - known_test_labels
)
if unknown_split_labels:
    raise ValueError(
        "Unrecognised split labels in the fixed split file: "
        f"{unknown_split_labels}"
    )

X_all_raw = (
    data[candidate_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)

X_train_values = X_all_raw.iloc[train_indices].to_numpy(dtype=float)
y_train = data[TARGET_COLUMN].iloc[train_indices].to_numpy(dtype=float)
groups_train = (
    data[GROUP_COLUMN]
    .iloc[train_indices]
    .astype(str)
    .to_numpy()
)

codes_train = data[CODE_COLUMN].iloc[train_indices].reset_index(drop=True)
candidate_metadata = candidate_table.set_index("feature")
candidate_families = np.asarray(
    [str(candidate_metadata.loc[feature, "family"]) for feature in candidate_features],
    dtype=object,
)
all_families = sorted(pd.unique(candidate_families).tolist())

n_train_groups = len(np.unique(groups_train))
if n_train_groups < N_INNER_SPLITS:
    raise ValueError(
        f"Only {n_train_groups} train groups; cannot use "
        f"{N_INNER_SPLITS}-fold GroupKFold."
    )

inner_splitter = GroupKFold(n_splits=N_INNER_SPLITS)
inner_splits = list(inner_splitter.split(X_train_values, y_train, groups_train))

split_table = pd.DataFrame(
    {
        CODE_COLUMN: data[CODE_COLUMN].astype(str),
        TARGET_COLUMN: data[TARGET_COLUMN],
        GROUP_COLUMN: data[GROUP_COLUMN].astype(str),
        "split": np.where(is_train, "Train", "Internal_evaluation"),
    }
)
split_table.to_csv(SPLIT_OUT, index=False)

print("=" * 96)
print("TG RANDOM-FOREST DIVERSITY-AWARE GA (OUTER-TRAIN ONLY)")
print("=" * 96)
print(f"Split source          : {SPLIT_IN.name}")
print(f"Candidate descriptors : {len(candidate_features)}")
print(f"Subset sizes          : {SUBSET_SIZES}")
print(f"Outer-train samples   : {len(y_train)}")
print(f"Outer-train groups    : {n_train_groups}")
print(f"Inner validation      : {N_INNER_SPLITS}-fold GroupKFold")
print(f"Outer test untouched  : {len(test_indices)} samples")
print(
    "RF search parameters  : "
    f"n_estimators={RF_GA_PARAMS['n_estimators']}, "
    f"max_depth={RF_GA_PARAMS['max_depth']}, "
    f"min_samples_split={RF_GA_PARAMS['min_samples_split']}, "
    f"min_samples_leaf={RF_GA_PARAMS['min_samples_leaf']}, "
    f"random_state={RF_GA_PARAMS['random_state']}"
)


# ============================================================
# 4. GA SWEEP ACROSS DIMENSIONALITY
# ============================================================

all_run_records = []
best_by_size = []
all_best_feature_records = []

for subset_size in SUBSET_SIZES:
    size_dir = OUTPUT_ROOT / f"subset_{subset_size}"
    size_dir.mkdir(parents=True, exist_ok=True)

    size_run_records = []
    best_size_chromosome = None
    best_size_result = None
    best_size_run = None
    all_histories = []

    for run_number in range(1, N_GA_RUNS + 1):
        print("\n" + "=" * 96)
        print(f"SUBSET {subset_size} | RF GA RUN {run_number}/{N_GA_RUNS}")
        print("=" * 96)

        chromosome, result, history = run_ga(
            subset_size=subset_size,
            run_number=run_number,
            X_values=X_train_values,
            y_values=y_train,
            groups=groups_train,
            inner_splits=inner_splits,
            candidate_families=candidate_families,
            all_families=all_families,
        )

        record = {
            "subset_size": subset_size,
            "run": run_number,
            "fitness": result.fitness,
            "oof_r2": result.oof_r2,
            "oof_rmse": result.oof_rmse,
            "oof_mae": result.oof_mae,
            "mean_fold_r2": result.mean_fold_r2,
            "sd_fold_r2": result.sd_fold_r2,
            "diversity_score": result.diversity_score,
            "concentration_penalty": result.concentration_penalty,
            "family_coverage": result.family_coverage,
            "family_entropy": result.family_entropy,
            "maximum_family_fraction": result.maximum_family_fraction,
            "n_families_present": result.n_families_present,
            "selected_feature_indices": json.dumps(list(chromosome)),
        }
        size_run_records.append(record)
        all_run_records.append(record)
        all_histories.extend(history)

        if (
            best_size_result is None
            or (result.fitness, result.oof_r2)
            > (best_size_result.fitness, best_size_result.oof_r2)
        ):
            best_size_chromosome = chromosome
            best_size_result = result
            best_size_run = run_number

    pd.DataFrame(size_run_records).to_csv(
        size_dir / "ga_run_summary.csv",
        index=False,
    )
    pd.DataFrame(all_histories).to_csv(
        size_dir / "ga_generation_history.csv",
        index=False,
    )

    selected_features = [
        candidate_features[index]
        for index in best_size_chromosome
    ]
    selected_table = candidate_table[
        candidate_table["feature"].isin(selected_features)
    ].copy()
    order_map = {
        feature: order + 1
        for order, feature in enumerate(selected_features)
    }
    selected_table["selection_order"] = selected_table["feature"].map(order_map)
    selected_table = selected_table.sort_values("selection_order").reset_index(drop=True)
    selected_table.to_csv(
        size_dir / "best_selected_descriptors.csv",
        index=False,
    )

    best_record = {
        "subset_size": subset_size,
        "best_run": best_size_run,
        "fitness": best_size_result.fitness,
        "oof_r2": best_size_result.oof_r2,
        "oof_rmse": best_size_result.oof_rmse,
        "oof_mae": best_size_result.oof_mae,
        "mean_fold_r2": best_size_result.mean_fold_r2,
        "sd_fold_r2": best_size_result.sd_fold_r2,
        "diversity_score": best_size_result.diversity_score,
        "concentration_penalty": best_size_result.concentration_penalty,
        "family_coverage": best_size_result.family_coverage,
        "family_entropy": best_size_result.family_entropy,
        "maximum_family_fraction": best_size_result.maximum_family_fraction,
        "n_families_present": best_size_result.n_families_present,
        "selected_descriptor_file": str(size_dir / "best_selected_descriptors.csv"),
    }
    best_by_size.append(best_record)

    selected_export = selected_table.copy()
    selected_export.insert(0, "subset_size", subset_size)
    selected_export.insert(1, "best_run", best_size_run)
    all_best_feature_records.append(selected_export)

    print("\nBest result for subset size", subset_size)
    print(f"  Best run : {best_size_run}")
    print(f"  OOF R2   : {best_size_result.oof_r2:.4f}")
    print(f"  OOF RMSE : {best_size_result.oof_rmse:.4f}")
    print(f"  OOF MAE  : {best_size_result.oof_mae:.4f}")
    print(f"  Fitness  : {best_size_result.fitness:.4f}")


# ============================================================
# 5. SAVE OVERALL RESULTS
# ============================================================

comparison = pd.DataFrame(best_by_size).sort_values("subset_size").reset_index(drop=True)
comparison.to_csv(OVERALL_COMPARISON_OUT, index=False)
pd.DataFrame(all_run_records).to_csv(ALL_RUNS_OUT, index=False)

if all_best_feature_records:
    pd.concat(all_best_feature_records, ignore_index=True).to_csv(
        OVERALL_BEST_FEATURES_OUT,
        index=False,
    )

recommended = comparison.sort_values(
    by=["fitness", "oof_r2", "oof_rmse"],
    ascending=[False, False, True],
).iloc[0]

print("\n" + "=" * 96)
print("TG RF DAGA SUBSET-SIZE COMPARISON COMPLETE")
print("=" * 96)
print(
    comparison[
        [
            "subset_size",
            "best_run",
            "oof_r2",
            "oof_rmse",
            "oof_mae",
            "fitness",
            "n_families_present",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(
    "\nRecommended dimensionality by inner group-OOF fitness: "
    f"{int(recommended['subset_size'])} descriptors"
)
print(f"OOF R2 = {recommended['oof_r2']:.4f}")
print(f"OOF RMSE = {recommended['oof_rmse']:.4f}")
print("\nSaved:")
for path in [
    SPLIT_OUT,
    OVERALL_COMPARISON_OUT,
    OVERALL_BEST_FEATURES_OUT,
    ALL_RUNS_OUT,
]:
    print(" -", path)
print("\n✅ Outer internal-evaluation samples were not used by GA.")

