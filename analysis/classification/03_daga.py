from __future__ import annotations

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


from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import gc
import json
import random
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. PATHS
# ============================================================
BASE = data_path('')
TRAIN_PATH = data_path('master_train_features.csv')
CANDIDATE_PATH = data_path('GA_candidate_descriptors_1000.csv')
SPLIT_IN = data_path('Final_clean_split_seed42.csv')
OUTPUT_ROOT = data_path('DiversityGA_SubsetSize_Comparison')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SPLIT_OUT = data_path('DiversityGA_SubsetSize_Comparison/fixed_train_test_split.csv')
OVERALL_COMPARISON_OUT = data_path('DiversityGA_SubsetSize_Comparison/subset_size_comparison.csv')
OVERALL_BEST_FEATURES_OUT = data_path('DiversityGA_SubsetSize_Comparison/all_subset_best_descriptors.csv')

# ============================================================
# 1. SETTINGS
# ============================================================
NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]
SEED = 42
N_SPLITS = 5
SUBSET_SIZES = [10, 20, 30, 40, 50, 60]

# Full experiment
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
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}
USE_CLASS_WEIGHT = True

FITNESS_WEIGHT_BALANCED_ACCURACY = 0.30
FITNESS_WEIGHT_MCC = 0.25
FITNESS_WEIGHT_PR_NON_NADES = 0.20
FITNESS_WEIGHT_ROC_AUC = 0.15
FITNESS_WEIGHT_DIVERSITY = 0.10
MAX_DESIRED_FAMILY_FRACTION = 0.45
FAMILY_CONCENTRATION_PENALTY_WEIGHT = 0.05

random.seed(SEED)
np.random.seed(SEED)

Chromosome = tuple[int, ...]


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    predictive_score: float
    diversity_score: float
    concentration_penalty: float
    roc_auc: float
    pr_auc_non_nades: float
    balanced_accuracy: float
    mcc: float
    threshold: float
    specificity_non_nades: float
    precision_nades: float
    recall_nades: float
    f1_nades: float
    family_coverage: float
    family_entropy: float
    orientation_balance: float
    maximum_family_fraction: float
    n_families_present: int
    n_NADES_oriented: int
    n_non_NADES_oriented: int


def validate_binary_target(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError("activity contains missing or non-numeric values.")
    unexpected = set(values.unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"activity must contain only 0 and 1: {unexpected}")
    return values.astype(np.int8).to_numpy()


def calculate_scale_pos_weight(y_values: np.ndarray) -> float:
    n0 = int(np.sum(y_values == 0))
    n1 = int(np.sum(y_values == 1))
    if n0 == 0 or n1 == 0:
        raise ValueError("Both classes must be present.")
    return n0 / n1 if USE_CLASS_WEIGHT else 1.0


def build_xgb_model(seed: int, spw: float) -> xgb.XGBClassifier:
    params = dict(XGB_PARAMS)
    params["random_state"] = seed
    return xgb.XGBClassifier(**params, scale_pos_weight=spw)


def calculate_threshold_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "specificity_non_nades": float(specificity),
        "precision_nades": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_nades": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_nades": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_balanced_threshold(y_true, probability):
    best_threshold, best_objective, best_metrics = 0.50, -np.inf, None
    for threshold in THRESHOLD_GRID:
        metrics = calculate_threshold_metrics(y_true, probability, float(threshold))
        scaled_mcc = (metrics["mcc"] + 1.0) / 2.0
        objective = 0.60 * metrics["balanced_accuracy"] + 0.40 * scaled_mcc
        if objective > best_objective:
            best_threshold = float(threshold)
            best_objective = float(objective)
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("Threshold optimization failed.")
    return best_threshold, best_metrics, best_objective


def calculate_probability_metrics(y_true, probability):
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_nades": float(average_precision_score(y_true, probability)),
        "pr_auc_non_nades": float(average_precision_score(1 - y_true, 1.0 - probability)),
    }


def create_prediction_table(codes, y_true, probability, threshold, split_name, subset_size):
    prediction = (probability >= threshold).astype(np.int8)
    return pd.DataFrame({
        "Code": codes.reset_index(drop=True),
        "subset_size": subset_size,
        "split": split_name,
        "true_activity": y_true,
        "true_class": np.where(y_true == 1, "NADES", "non-NADES"),
        "probability_NADES": probability,
        "probability_non_NADES": 1.0 - probability,
        "threshold": threshold,
        "predicted_activity": prediction,
        "predicted_class": np.where(prediction == 1, "NADES", "non-NADES"),
        "correct_prediction": prediction == y_true,
    })


# ============================================================
# 2. LOAD DATA
# ============================================================
for required_path in [TRAIN_PATH, CANDIDATE_PATH, SPLIT_IN]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found:\n{required_path}")

data = pd.read_csv(TRAIN_PATH, na_values=NA_VALUES, keep_default_na=True, low_memory=False)
candidate_table = pd.read_csv(CANDIDATE_PATH, low_memory=False)

if not {"Code", "activity"}.issubset(data.columns):
    raise ValueError("Dataset must contain Code and activity columns.")
if not {"feature", "family", "orientation"}.issubset(candidate_table.columns):
    raise ValueError("Candidate table must contain feature, family, and orientation columns.")

candidate_table = candidate_table.dropna(subset=["feature"]).copy()
candidate_table["feature"] = candidate_table["feature"].astype(str).str.strip()
candidate_table = candidate_table.drop_duplicates("feature", keep="first").reset_index(drop=True)
candidate_features = candidate_table["feature"].tolist()

if len(candidate_features) < max(SUBSET_SIZES):
    raise ValueError("Candidate pool is smaller than the largest requested subset size.")

missing_in_data = [f for f in candidate_features if f not in data.columns]
if missing_in_data:
    raise ValueError(f"Candidate descriptors absent from dataset: {missing_in_data[:20]}")

y_all = validate_binary_target(data["activity"])
codes_all = data["Code"].astype("string").str.strip()
if codes_all.isna().any() or codes_all.duplicated().any():
    raise ValueError("Code values must be non-missing and unique.")

X_all_raw = data[candidate_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
candidate_metadata = candidate_table.set_index("feature")
candidate_families = np.asarray([str(candidate_metadata.loc[f, "family"]) for f in candidate_features], dtype=object)
candidate_orientations = np.asarray([str(candidate_metadata.loc[f, "orientation"]) for f in candidate_features], dtype=object)
ALL_FAMILIES = sorted(pd.unique(candidate_families).tolist())

# ============================================================
# 3. FIXED SPLIT — reuse the prefilter split (match by Code)
#    The held-out (internal-evaluation) subset is NOT touched here.
#    It is reserved for a separate final-evaluation script.
# ============================================================
split_ref = pd.read_csv(SPLIT_IN)
if not {"Code", "split"}.issubset(split_ref.columns):
    raise ValueError("Split file must contain 'Code' and 'split' columns.")

split_ref["Code"] = split_ref["Code"].astype("string").str.strip()
split_map = dict(zip(split_ref["Code"], split_ref["split"]))

split_label = codes_all.map(split_map)
if split_label.isna().any():
    missing = codes_all[split_label.isna()].tolist()[:20]
    raise ValueError(
        f"{int(split_label.isna().sum())} codes are absent from the split file. "
        f"Examples: {missing}"
    )

is_train = (split_label == "Train").to_numpy()
train_indices = np.where(is_train)[0]

if len(train_indices) == 0:
    raise ValueError("Split produced an empty training partition.")

# Only the training partition is loaded into memory. Held-out rows are dropped.
X_train_raw = X_all_raw.iloc[train_indices].reset_index(drop=True)
y_train = y_all[train_indices]
codes_train = codes_all.iloc[train_indices].reset_index(drop=True)

split_table = pd.DataFrame({
    "Code": codes_all,
    "activity": y_all,
    "class": np.where(y_all == 1, "NADES", "non-NADES"),
    "split": np.where(is_train, "Train", "Internal_evaluation"),
})
split_table.to_csv(SPLIT_OUT, index=False)

print("=" * 110)
print("DIVERSITY-AWARE GA SUBSET-SIZE COMPARISON  (OOF-only; held-out untouched)")
print("=" * 110)
print("Split source :", SPLIT_IN.name)
print("Candidate descriptors:", len(candidate_features))
print("Subset sizes:", SUBSET_SIZES)
print("Train samples:", len(train_indices))
print("Train class balance:", dict(zip(*np.unique(y_train, return_counts=True))))

# ============================================================
# 4. INNER-CV PREPROCESSING (train only)
# ============================================================
inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
CV_SPLITS = list(inner_cv.split(X_train_raw, y_train))
FOLD_DATA = []
for fold_number, (inner_train_index, inner_validation_index) in enumerate(CV_SPLITS, start=1):
    fold_imputer = SimpleImputer(strategy="median")
    X_fold_train = fold_imputer.fit_transform(X_train_raw.iloc[inner_train_index]).astype(np.float32)
    X_fold_validation = fold_imputer.transform(X_train_raw.iloc[inner_validation_index]).astype(np.float32)
    FOLD_DATA.append({
        "fold_number": fold_number,
        "train_index": inner_train_index,
        "validation_index": inner_validation_index,
        "X_train": X_fold_train,
        "X_validation": X_fold_validation,
        "y_train": y_train[inner_train_index],
        "y_validation": y_train[inner_validation_index],
        "spw": calculate_scale_pos_weight(y_train[inner_train_index]),
    })


def normalized_shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    maximum_entropy = np.log(len(counts))
    return float(entropy / maximum_entropy) if maximum_entropy > 0 else 0.0


def calculate_descriptor_diversity(chromosome, subset_size):
    indices = np.asarray(chromosome, dtype=int)
    selected_families = candidate_families[indices]
    selected_orientations = candidate_orientations[indices]
    family_counts = {family: int(np.sum(selected_families == family)) for family in ALL_FAMILIES}
    family_count_array = np.asarray([family_counts[f] for f in ALL_FAMILIES], dtype=float)
    n_families_present = int(np.sum(family_count_array > 0))
    family_coverage = n_families_present / len(ALL_FAMILIES) if ALL_FAMILIES else 0.0
    family_entropy = normalized_shannon_entropy(family_count_array)
    maximum_family_fraction = float(family_count_array.max() / subset_size)

    n_nades_oriented = int(np.sum(selected_orientations == "NADES_oriented"))
    n_non_nades_oriented = int(np.sum(selected_orientations == "non_NADES_oriented"))
    n_directional = n_nades_oriented + n_non_nades_oriented
    if n_directional > 0:
        nades_fraction = n_nades_oriented / n_directional
        orientation_balance = float(np.clip(1.0 - 2.0 * abs(nades_fraction - 0.5), 0.0, 1.0))
    else:
        orientation_balance = 0.0

    diversity_score = 0.40 * family_coverage + 0.40 * family_entropy + 0.20 * orientation_balance
    concentration_excess = max(0.0, maximum_family_fraction - MAX_DESIRED_FAMILY_FRACTION)
    concentration_penalty = concentration_excess / max(1.0 - MAX_DESIRED_FAMILY_FRACTION, 1e-12)

    return {
        "diversity_score": float(diversity_score),
        "family_coverage": float(family_coverage),
        "family_entropy": float(family_entropy),
        "orientation_balance": float(orientation_balance),
        "maximum_family_fraction": float(maximum_family_fraction),
        "concentration_penalty": float(concentration_penalty),
        "n_families_present": n_families_present,
        "n_NADES_oriented": n_nades_oriented,
        "n_non_NADES_oriented": n_non_nades_oriented,
        "family_counts": family_counts,
    }


def canonicalize_chromosome(chromosome, subset_size):
    unique_indices = sorted({int(index) for index in chromosome})
    if len(unique_indices) != subset_size:
        raise ValueError(f"Chromosome must contain exactly {subset_size} unique indices.")
    if min(unique_indices) < 0 or max(unique_indices) >= len(candidate_features):
        raise ValueError("Chromosome contains an out-of-range index.")
    return tuple(unique_indices)


def create_random_chromosome(rng, subset_size):
    return canonicalize_chromosome(
        rng.choice(len(candidate_features), size=subset_size, replace=False),
        subset_size,
    )


def initialize_population(rng, subset_size):
    population, seen = [], set()
    while len(population) < POPULATION_SIZE:
        chromosome = create_random_chromosome(rng, subset_size)
        if chromosome not in seen:
            population.append(chromosome)
            seen.add(chromosome)
    return population


def tournament_selection(population, fitness_results, rng):
    positions = rng.choice(len(population), size=TOURNAMENT_SIZE, replace=False)
    tournament = [population[position] for position in positions]
    return max(tournament, key=lambda chromosome: fitness_results[chromosome].fitness)


def crossover(parent_1, parent_2, rng, subset_size):
    set_1, set_2 = set(parent_1), set(parent_2)
    shared, union = set_1 & set_2, list(set_1 | set_2)

    def create_child():
        child = set(shared)
        available_union = [index for index in union if index not in child]
        rng.shuffle(available_union)
        child.update(available_union[: subset_size - len(child)])
        if len(child) < subset_size:
            available_global = np.setdiff1d(np.arange(len(candidate_features)), np.asarray(list(child), dtype=int))
            child.update(rng.choice(available_global, size=subset_size - len(child), replace=False).tolist())
        return canonicalize_chromosome(child, subset_size)

    return create_child(), create_child()


def mutate(chromosome, rng, subset_size):
    selected = set(chromosome)
    n_replacements = min(
        int(rng.integers(MIN_MUTATION_REPLACEMENTS, MAX_MUTATION_REPLACEMENTS + 1)),
        subset_size,
    )
    remove_indices = rng.choice(np.asarray(list(selected), dtype=int), size=n_replacements, replace=False)
    selected.difference_update(remove_indices.tolist())
    available = np.setdiff1d(np.arange(len(candidate_features)), np.asarray(list(selected), dtype=int))
    selected.update(rng.choice(available, size=n_replacements, replace=False).tolist())
    return canonicalize_chromosome(selected, subset_size)


def run_subset_size_experiment(subset_size):
    subset_dir = OUTPUT_ROOT / f"subset_{subset_size:02d}"
    subset_dir.mkdir(parents=True, exist_ok=True)
    fitness_cache = {}

    def evaluate_chromosome(chromosome):
        chromosome = canonicalize_chromosome(chromosome, subset_size)
        if chromosome in fitness_cache:
            return fitness_cache[chromosome]

        selected_indices = np.asarray(chromosome, dtype=int)
        oof_probability = np.full(len(y_train), np.nan, dtype=float)

        for fold in FOLD_DATA:
            model = build_xgb_model(
                SEED + subset_size * 100 + fold["fold_number"],
                fold["spw"],
            )
            model.fit(fold["X_train"][:, selected_indices], fold["y_train"])
            oof_probability[fold["validation_index"]] = model.predict_proba(
                fold["X_validation"][:, selected_indices]
            )[:, 1]
            del model

        probability_metrics = calculate_probability_metrics(y_train, oof_probability)
        threshold, threshold_metrics, _ = find_balanced_threshold(y_train, oof_probability)
        scaled_mcc = (threshold_metrics["mcc"] + 1.0) / 2.0
        diversity_metrics = calculate_descriptor_diversity(chromosome, subset_size)

        predictive_score = (
            FITNESS_WEIGHT_BALANCED_ACCURACY * threshold_metrics["balanced_accuracy"]
            + FITNESS_WEIGHT_MCC * scaled_mcc
            + FITNESS_WEIGHT_PR_NON_NADES * probability_metrics["pr_auc_non_nades"]
            + FITNESS_WEIGHT_ROC_AUC * probability_metrics["roc_auc"]
        )
        fitness = (
            predictive_score
            + FITNESS_WEIGHT_DIVERSITY * diversity_metrics["diversity_score"]
            - FAMILY_CONCENTRATION_PENALTY_WEIGHT * diversity_metrics["concentration_penalty"]
        )

        result = FitnessResult(
            fitness=float(fitness),
            predictive_score=float(predictive_score),
            diversity_score=float(diversity_metrics["diversity_score"]),
            concentration_penalty=float(diversity_metrics["concentration_penalty"]),
            roc_auc=float(probability_metrics["roc_auc"]),
            pr_auc_non_nades=float(probability_metrics["pr_auc_non_nades"]),
            balanced_accuracy=float(threshold_metrics["balanced_accuracy"]),
            mcc=float(threshold_metrics["mcc"]),
            threshold=float(threshold),
            specificity_non_nades=float(threshold_metrics["specificity_non_nades"]),
            precision_nades=float(threshold_metrics["precision_nades"]),
            recall_nades=float(threshold_metrics["recall_nades"]),
            f1_nades=float(threshold_metrics["f1_nades"]),
            family_coverage=float(diversity_metrics["family_coverage"]),
            family_entropy=float(diversity_metrics["family_entropy"]),
            orientation_balance=float(diversity_metrics["orientation_balance"]),
            maximum_family_fraction=float(diversity_metrics["maximum_family_fraction"]),
            n_families_present=int(diversity_metrics["n_families_present"]),
            n_NADES_oriented=int(diversity_metrics["n_NADES_oriented"]),
            n_non_NADES_oriented=int(diversity_metrics["n_non_NADES_oriented"]),
        )
        fitness_cache[chromosome] = result
        return result

    def run_one_ga(run_number):
        rng = np.random.default_rng(SEED + subset_size * 10000 + 1000 * run_number)
        population = initialize_population(rng, subset_size)
        run_history = []
        best_chromosome = None
        best_result = None
        no_improvement = 0

        print("\n" + "=" * 110)
        print(f"SUBSET {subset_size} | GA RUN {run_number}/{N_GA_RUNS}")
        print("=" * 110)

        for generation in range(1, N_GENERATIONS + 1):
            start = time.perf_counter()
            fitness_results = {chromosome: evaluate_chromosome(chromosome) for chromosome in population}
            ranked_population = sorted(population, key=lambda chromosome: fitness_results[chromosome].fitness, reverse=True)
            generation_best = ranked_population[0]
            generation_best_result = fitness_results[generation_best]

            if best_result is None or generation_best_result.fitness > best_result.fitness + FITNESS_TOLERANCE:
                best_chromosome = generation_best
                best_result = generation_best_result
                no_improvement = 0
            else:
                no_improvement += 1

            values = np.asarray([fitness_results[c].fitness for c in population], dtype=float)
            elapsed = time.perf_counter() - start
            run_history.append({
                "subset_size": subset_size,
                "run": run_number,
                "generation": generation,
                "best_fitness_generation": generation_best_result.fitness,
                "best_fitness_run": best_result.fitness,
                "mean_fitness": float(values.mean()),
                "std_fitness": float(values.std(ddof=1)),
                "best_roc_auc": best_result.roc_auc,
                "best_pr_auc_non_nades": best_result.pr_auc_non_nades,
                "best_balanced_accuracy": best_result.balanced_accuracy,
                "best_mcc": best_result.mcc,
                "best_threshold": best_result.threshold,
                "generations_without_improvement": no_improvement,
                "fitness_cache_size": len(fitness_cache),
                "seconds": elapsed,
            })

            print(
                f"Generation {generation:02d} | fitness={best_result.fitness:.5f} "
                f"| ROC={best_result.roc_auc:.4f} | PR0={best_result.pr_auc_non_nades:.4f} "
                f"| BA={best_result.balanced_accuracy:.4f} | MCC={best_result.mcc:.4f} "
                f"| threshold={best_result.threshold:.2f} | no improvement={no_improvement}"
            )

            if no_improvement >= EARLY_STOPPING_GENERATIONS:
                print("Early stopping.")
                break

            next_population = ranked_population[:ELITE_COUNT].copy()
            next_population_set = set(next_population)
            while len(next_population) < POPULATION_SIZE:
                parent_1 = tournament_selection(population, fitness_results, rng)
                parent_2 = tournament_selection(population, fitness_results, rng)
                if rng.random() < CROSSOVER_PROBABILITY:
                    child_1, child_2 = crossover(parent_1, parent_2, rng, subset_size)
                else:
                    child_1, child_2 = parent_1, parent_2

                for child in [child_1, child_2]:
                    if rng.random() < MUTATION_PROBABILITY:
                        child = mutate(child, rng, subset_size)
                    attempts = 0
                    while child in next_population_set and attempts < 10:
                        child = mutate(child, rng, subset_size)
                        attempts += 1
                    if child not in next_population_set:
                        next_population.append(child)
                        next_population_set.add(child)
                    if len(next_population) >= POPULATION_SIZE:
                        break
            population = next_population
            gc.collect()

        if best_chromosome is None or best_result is None:
            raise RuntimeError(f"GA run {run_number} failed.")
        return {
            "run": run_number,
            "best_chromosome": best_chromosome,
            "best_result": best_result,
            "history": run_history,
        }

    all_run_results, all_history_records = [], []
    subset_start = time.perf_counter()
    for run_number in range(1, N_GA_RUNS + 1):
        run_result = run_one_ga(run_number)
        all_run_results.append(run_result)
        all_history_records.extend(run_result["history"])
    subset_elapsed = time.perf_counter() - subset_start

    run_summary_records = []
    selection_counter = Counter()
    for run_result in all_run_results:
        chromosome = run_result["best_chromosome"]
        result = run_result["best_result"]
        feature_names = [candidate_features[index] for index in chromosome]
        diversity = calculate_descriptor_diversity(chromosome, subset_size)
        run_summary_records.append({
            "subset_size": subset_size,
            "run": run_result["run"],
            "fitness": result.fitness,
            "predictive_score": result.predictive_score,
            "diversity_score": result.diversity_score,
            "oof_roc_auc": result.roc_auc,
            "oof_pr_auc_non_nades": result.pr_auc_non_nades,
            "oof_balanced_accuracy": result.balanced_accuracy,
            "oof_mcc": result.mcc,
            "oof_threshold": result.threshold,
            "family_counts_json": json.dumps(diversity["family_counts"]),
            "feature_names_json": json.dumps(feature_names),
        })
        for feature_index in chromosome:
            selection_counter[feature_index] += 1

    pd.DataFrame(run_summary_records).sort_values("fitness", ascending=False).to_csv(subset_dir / "GA_runs.csv", index=False)
    pd.DataFrame(all_history_records).to_csv(subset_dir / "GA_history.csv", index=False)

    stability_records = []
    for feature_index, count in selection_counter.items():
        feature = candidate_features[feature_index]
        record = {
            "subset_size": subset_size,
            "feature_index": feature_index,
            "feature": feature,
            "selected_runs": count,
            "selection_frequency": count / N_GA_RUNS,
        }
        if feature in candidate_metadata.index:
            for column in [
                "family", "orientation", "cohens_d", "abs_cohens_d",
                "mutual_information", "orientation_free_auc",
                "mann_whitney_fdr", "informativeness_index",
            ]:
                if column in candidate_metadata.columns:
                    record[column] = candidate_metadata.loc[feature, column]
        stability_records.append(record)
    pd.DataFrame(stability_records).sort_values("selection_frequency", ascending=False).to_csv(subset_dir / "descriptor_stability.csv", index=False)

    best_run_result = max(all_run_results, key=lambda item: item["best_result"].fitness)
    best_chromosome = best_run_result["best_chromosome"]
    best_ga_result = best_run_result["best_result"]
    best_feature_indices = np.asarray(best_chromosome, dtype=int)
    best_features = [candidate_features[index] for index in best_chromosome]
    best_diversity = calculate_descriptor_diversity(best_chromosome, subset_size)

    best_feature_table = pd.DataFrame({
        "subset_size": subset_size,
        "selected_order": np.arange(1, len(best_features) + 1),
        "candidate_index": best_feature_indices,
        "feature": best_features,
    }).merge(candidate_table, on="feature", how="left")
    best_feature_table.to_csv(subset_dir / "best_selected_descriptors.csv", index=False)

    family_summary_df = pd.DataFrame([
        {
            "subset_size": subset_size,
            "family": family,
            "selected_count": int(best_diversity["family_counts"].get(family, 0)),
            "selected_fraction": best_diversity["family_counts"].get(family, 0) / subset_size,
        }
        for family in ALL_FAMILIES
    ]).sort_values("selected_count", ascending=False)
    family_summary_df.to_csv(subset_dir / "best_family_composition.csv", index=False)

    # OOF probabilities for the best chromosome (train only)
    best_oof_probability = np.full(len(y_train), np.nan, dtype=float)
    best_oof_fold = np.full(len(y_train), -1, dtype=int)
    for fold in FOLD_DATA:
        model = build_xgb_model(SEED + subset_size * 100 + fold["fold_number"], fold["spw"])
        model.fit(fold["X_train"][:, best_feature_indices], fold["y_train"])
        best_oof_probability[fold["validation_index"]] = model.predict_proba(
            fold["X_validation"][:, best_feature_indices]
        )[:, 1]
        best_oof_fold[fold["validation_index"]] = fold["fold_number"]
        del model

    best_threshold, best_oof_threshold_metrics, best_threshold_objective = find_balanced_threshold(y_train, best_oof_probability)
    best_oof_probability_metrics = calculate_probability_metrics(y_train, best_oof_probability)
    best_oof_output = create_prediction_table(
        codes_train, y_train, best_oof_probability, best_threshold, "Train_OOF", subset_size
    )
    best_oof_output["fold"] = best_oof_fold
    best_oof_output.to_csv(subset_dir / "best_model_oof_predictions.csv", index=False)

    threshold_records = []
    for threshold in THRESHOLD_GRID:
        metrics = calculate_threshold_metrics(y_train, best_oof_probability, float(threshold))
        scaled_mcc = (metrics["mcc"] + 1.0) / 2.0
        objective = 0.60 * metrics["balanced_accuracy"] + 0.40 * scaled_mcc
        threshold_records.append({
            "subset_size": subset_size,
            "threshold": float(threshold),
            "threshold_objective": float(objective),
            **metrics,
        })
    pd.DataFrame(threshold_records).to_csv(subset_dir / "threshold_scan.csv", index=False)

    # Persist the selected descriptor names for this subset size (for the
    # separate final-evaluation script to load once a size is chosen).
    with open(subset_dir / "selected_features.txt", "w", encoding="utf-8") as fh:
        for feature in best_features:
            fh.write(feature + "\n")

    summary = {
        "subset_size": subset_size,
        "best_GA_run": best_run_result["run"],
        "GA_total_fitness": best_ga_result.fitness,
        "GA_predictive_score": best_ga_result.predictive_score,
        "GA_diversity_score": best_ga_result.diversity_score,
        "GA_concentration_penalty": best_ga_result.concentration_penalty,
        "family_coverage": best_diversity["family_coverage"],
        "family_entropy": best_diversity["family_entropy"],
        "orientation_balance": best_diversity["orientation_balance"],
        "maximum_family_fraction": best_diversity["maximum_family_fraction"],
        "OOF_threshold": best_threshold,
        "OOF_threshold_objective": best_threshold_objective,
        "OOF_ROC_AUC": best_oof_probability_metrics["roc_auc"],
        "OOF_PR_AUC_NADES": best_oof_probability_metrics["pr_auc_nades"],
        "OOF_PR_AUC_non_NADES": best_oof_probability_metrics["pr_auc_non_nades"],
        "OOF_balanced_accuracy": best_oof_threshold_metrics["balanced_accuracy"],
        "OOF_MCC": best_oof_threshold_metrics["mcc"],
        "OOF_specificity_non_NADES": best_oof_threshold_metrics["specificity_non_nades"],
        "OOF_recall_NADES": best_oof_threshold_metrics["recall_nades"],
        "population_size": POPULATION_SIZE,
        "maximum_generations": N_GENERATIONS,
        "GA_runs": N_GA_RUNS,
        "fitness_cache_size": len(fitness_cache),
        "total_seconds": subset_elapsed,
    }
    pd.DataFrame([summary]).to_csv(subset_dir / "subset_summary.csv", index=False)

    print("\n" + "-" * 110)
    print(f"SUBSET SIZE {subset_size} COMPLETED")
    print("-" * 110)
    print(f"OOF ROC-AUC           : {summary['OOF_ROC_AUC']:.4f}")
    print(f"OOF PR-AUC non-NADES  : {summary['OOF_PR_AUC_non_NADES']:.4f}")
    print(f"OOF balanced accuracy : {summary['OOF_balanced_accuracy']:.4f}")
    print(f"OOF MCC               : {summary['OOF_MCC']:.4f}")
    print(f"OOF specificity       : {summary['OOF_specificity_non_NADES']:.4f}")
    print(f"GA total fitness      : {summary['GA_total_fitness']:.4f}")
    print(f"Selected threshold    : {best_threshold:.2f}")

    return summary, best_feature_table


# ============================================================
# 5. RUN ALL SUBSET SIZES
# ============================================================
overall_summaries = []
all_best_feature_tables = []
overall_start = time.perf_counter()

for subset_size in SUBSET_SIZES:
    summary, best_feature_table = run_subset_size_experiment(subset_size)
    overall_summaries.append(summary)
    all_best_feature_tables.append(best_feature_table)

overall_elapsed = time.perf_counter() - overall_start

comparison_df = pd.DataFrame(overall_summaries).sort_values("subset_size").reset_index(drop=True)
comparison_df["delta_OOF_ROC_from_best"] = comparison_df["OOF_ROC_AUC"].max() - comparison_df["OOF_ROC_AUC"]
comparison_df["delta_OOF_BA_from_best"] = comparison_df["OOF_balanced_accuracy"].max() - comparison_df["OOF_balanced_accuracy"]
comparison_df["delta_OOF_MCC_from_best"] = comparison_df["OOF_MCC"].max() - comparison_df["OOF_MCC"]
comparison_df.to_csv(OVERALL_COMPARISON_OUT, index=False)
pd.concat(all_best_feature_tables, ignore_index=True).to_csv(OVERALL_BEST_FEATURES_OUT, index=False)

columns = [
    "subset_size", "OOF_ROC_AUC", "OOF_PR_AUC_non_NADES", "OOF_balanced_accuracy",
    "OOF_MCC", "OOF_specificity_non_NADES", "OOF_threshold",
    "delta_OOF_ROC_from_best", "GA_total_fitness",
]

print("\n" + "=" * 130)
print("FINAL SUBSET-SIZE COMPARISON  (OOF only — dimensionality selection)")
print("=" * 130)
print(comparison_df[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nSaved:")
print(" -", SPLIT_OUT)
print(" -", OVERALL_COMPARISON_OUT)
print(" -", OVERALL_BEST_FEATURES_OUT)
print(" -", OUTPUT_ROOT)
print(f"\nTotal runtime: {overall_elapsed / 60.0:.2f} min")
print("\nSelect the smallest subset on the OOF performance plateau.")
print("The held-out (internal-evaluation) and blind sets remain reserved for the")
print("separate final-evaluation script and are NOT used for dimensionality selection.")


