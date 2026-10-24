"""Nested leave-one-component-out (LOCO) validation for NADES classification.

For every held-out component, *all* label-dependent operations are repeated
using only mixtures that do not contain that component:
    QC -> supervised screening -> ranking -> correlation pruning ->
    diversity-aware genetic algorithm -> inner-OOF threshold selection -> fit.

Consequently, no label belonging to a mixture with the held-out component can
influence its feature set, operating threshold, preprocessing, or model.

Input files required
--------------------
master_train_features.csv : Code, activity, and the full raw descriptor matrix
original.csv              : Code, A_id, B_id, C_id, D_id

The script deliberately does NOT read blind data.

Important computational note
----------------------------
This is much more expensive than fixed-Final60 LOCO. First use a small
OUTER_COMPONENTS pilot. For the definitive analysis, set it to None and retain
all eligible components, while recording the exact configuration below.
"""

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
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

BASE = data_path('')
MASTER_PATH = data_path('master_train_features.csv')
ORIGINAL_PATH = data_path('original.csv')
# Full, encoding-level metadata table. It must map every raw descriptor to the
# exact family used in the main DAGA pipeline. Orientation is label-derived
# and is recomputed within every outer LOCO fold.
# Required columns: feature, family.
FEATURE_METADATA_PATH = data_path('all_descriptor_metadata.csv')
OUTPUT_DIR = data_path('NestedLOCO_Final60')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_COLUMNS = ["A_id", "B_id", "C_id", "D_id"]
NA_VALUES = ["na", "NA", "Na", "N/A", "n/a", "None", "none", "-", "", "inf", "-inf", "Inf", "+inf"]

# Outer LOCO eligibility. Metrics requiring both classes are reported only for
# those outer folds, but one-class components may still be fitted if desired.
MIN_OUTER_TEST_SAMPLES = 5
REQUIRE_BOTH_OUTER_CLASSES = True

# Set e.g. {"Cholinechloride", "Maltose"} for a pilot. Set None for all
# eligible components in the definitive run.
OUTER_COMPONENTS: set[str] | None = None

# Selection architecture. Keeping FINAL_SUBSET_SIZE=60 makes this a direct
# nested test of the published Final60 architecture; the identities of the 60
# features are nevertheless reselected independently in every outer fold.
FINAL_SUBSET_SIZE = 60
TOP_RANKED = 4000
MAX_CANDIDATE_POOL = 1000
CORRELATION_LIMIT = 0.95

# Eligibility and composite-ranking constants. These are the frozen settings
# from the main classification pipeline and must not be changed for nested
# LOCO. They are never tuned on outer folds.
EFFECT_SIZE_MIN = 0.20
AUC_ORIENTATION_FREE_MIN = 0.55
FDR_Q_MAX = 0.05
MI_MIN = 1.0e-12
RANK_WEIGHTS = {"effect": 0.35, "mi": 0.30, "auc": 0.25, "significance": 0.10}

# Final DAGA objective, identical to the main classification pipeline:
# Fitness = P + 0.10 * Div - 0.05 * normalized_concentration_penalty.
# PP is the published balance of BAcc, rescaled MCC, minority-class PR-AUC,
# and ROC-AUC. Div uses descriptor-family coverage, normalized entropy, and
# A/B orientation balance.
PREDICTIVE_WEIGHTS = {"bacc": 0.30, "mcc": 0.25, "ap_non_nades": 0.20, "roc_auc": 0.15}
DIVERSITY_WEIGHTS = {"coverage": 0.40, "entropy": 0.40, "orientation": 0.20}
FITNESS_DIVERSITY_WEIGHT = 0.10
MAX_FAMILY_SHARE = 0.45
CONCENTRATION_PENALTY_WEIGHT = 0.05

# Genetic algorithm settings. These match the existing DAGA search as closely
# as possible. Reduce only for an explicitly labelled pilot run.
N_GA_RUNS = 5
GA_POPULATION = 15
GA_MAX_GENERATIONS = 8
GA_TOURNAMENT_SIZE = 3
GA_ELITE = 3
GA_CROSSOVER_PROBABILITY = 0.80
GA_MUTATION_PROBABILITY = 0.35
GA_STALL_GENERATIONS = 5
GA_MIN_IMPROVEMENT = 1.0e-5

# Inner validation is used only inside an outer training partition.
INNER_SPLITS = 5
INNER_SEED = 42
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)

# Main pipeline seed. Model seeds and GA-run seeds below follow its formulas.
MODEL_SEED = 42
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


# =============================================================================
# 2. DATA CLASSES AND LOW-LEVEL HELPERS
# =============================================================================

@dataclass
class SelectionResult:
    features: list[str]
    fitness: float
    n_after_qc: int
    n_eligible: int
    n_candidate_pool: int


def class_weight(y: np.ndarray) -> float:
    n_negative = int((y == 0).sum())
    n_positive = int((y == 1).sum())
    if not n_negative or not n_positive:
        raise ValueError("Training fold contains a single class.")
    return n_negative / n_positive


def build_model(y_train: np.ndarray, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **XGB_PARAMS,
        random_state=seed,
        scale_pos_weight=class_weight(y_train),
    )


def orientation_free_auc(y: np.ndarray, x: np.ndarray) -> float:
    auc = roc_auc_score(y, x)
    return max(auc, 1.0 - auc)


def percentile(values: np.ndarray) -> np.ndarray:
    """Percentile ranks in [0, 1], robust to ties and non-finite values."""
    series = pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def technical_qc(X: pd.DataFrame) -> pd.DataFrame:
    """Perform all label-free QC only on the current outer-training matrix."""
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().any(axis=0)
    X = X.loc[:, keep]
    keep = X.isna().mean(axis=0) <= 0.95
    X = X.loc[:, keep]
    keep = X.nunique(dropna=True) > 1
    X = X.loc[:, keep]

    # Exact duplicate detection includes NaN positions and values. The matrix is
    # moderate enough after initial QC; transpose avoids an error-prone hash
    # shortcut and is deterministic.
    X = X.T.drop_duplicates(keep="first").T
    return X


def supervised_ranking(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return imputed training data and a sorted table of eligible descriptors."""
    imputer = SimpleImputer(strategy="median")
    X_values = imputer.fit_transform(X_train)
    X = pd.DataFrame(X_values, index=X_train.index, columns=X_train.columns)

    y0 = y_train == 0
    y1 = y_train == 1
    n0, n1 = int(y0.sum()), int(y1.sum())
    mean0, mean1 = X.loc[y0].mean(axis=0), X.loc[y1].mean(axis=0)
    var0, var1 = X.loc[y0].var(axis=0, ddof=1), X.loc[y1].var(axis=0, ddof=1)
    pooled_sd = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / max(n0 + n1 - 2, 1))
    signed_effect = (mean1 - mean0).to_numpy(dtype=float)
    cohen_d = np.abs(signed_effect) / pooled_sd.replace(0, np.nan).to_numpy(dtype=float)
    cohen_d = np.nan_to_num(cohen_d, nan=0.0, posinf=0.0, neginf=0.0)

    # MI is stochastic; its seed is fixed. It is evaluated only on this fold.
    mi = mutual_info_classif(X.to_numpy(), y_train, random_state=INNER_SEED)
    auc = np.array([orientation_free_auc(y_train, X.iloc[:, j].to_numpy()) for j in range(X.shape[1])])

    p_values = np.empty(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        try:
            p_values[j] = mannwhitneyu(
                X.iloc[y0, j], X.iloc[y1, j], alternative="two-sided"
            ).pvalue
        except ValueError:
            p_values[j] = 1.0
    q_values = multipletests(p_values, method="fdr_bh")[1]
    significance = -np.log10(np.clip(q_values, 1.0e-300, 1.0))

    eligible = (
        (cohen_d >= EFFECT_SIZE_MIN)
        | (auc >= AUC_ORIENTATION_FREE_MIN)
        | ((mi > MI_MIN) & (q_values <= FDR_Q_MAX))
    )
    table = pd.DataFrame({
        "feature": X.columns,
        "effect": cohen_d,
        "orientation": np.where(
            signed_effect >= 0.0, "NADES_oriented", "non_NADES_oriented"
        ),
        "mi": mi,
        "auc": auc,
        "q_value": q_values,
        "significance": significance,
        "eligible": eligible,
    })
    table = table.loc[table["eligible"]].copy()
    if len(table) < FINAL_SUBSET_SIZE:
        raise RuntimeError(f"Only {len(table)} eligible descriptors; need {FINAL_SUBSET_SIZE}.")

    table["score"] = (
        RANK_WEIGHTS["effect"] * percentile(table["effect"].to_numpy())
        + RANK_WEIGHTS["mi"] * percentile(table["mi"].to_numpy())
        + RANK_WEIGHTS["auc"] * percentile((table["auc"] - 0.50).to_numpy())
        + RANK_WEIGHTS["significance"] * percentile(table["significance"].to_numpy())
    )
    return X, table.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


def correlation_prune(X: pd.DataFrame, ranked: pd.DataFrame) -> list[str]:
    """Greedy correlation pruning, retaining only higher-ranked descriptors."""
    shortlist = ranked.head(TOP_RANKED)["feature"].tolist()
    kept: list[str] = []
    for feature in shortlist:
        if not kept:
            kept.append(feature)
            continue
        correlations = X.loc[:, kept].corrwith(X[feature]).abs()
        if bool((correlations < CORRELATION_LIMIT).all()):
            kept.append(feature)
        if len(kept) >= MAX_CANDIDATE_POOL:
            break
    if len(kept) < FINAL_SUBSET_SIZE:
        raise RuntimeError(f"Correlation pruning retained only {len(kept)} descriptors.")
    return kept


def load_feature_metadata(path: Path, all_features: Iterable[str]) -> pd.DataFrame:
    """Load or deterministically build static descriptor-family labels.

    The master feature matrix has a controlled encoding grammar: A_/B_/C_/D_
    are component slots; mix_ is the stoichiometry-weighted family; avg_ is
    the arithmetic-mean family; AB_ contains pairwise operators; and the six
    remaining ratio/presence columns are composition terms. This mapping uses
    only feature names created by the encoding code and never labels.
    """
    if not path.exists():
        records = []
        for feature in all_features:
            if feature.startswith("A_"):
                family = "Component_A"
            elif feature.startswith("B_"):
                family = "Component_B"
            elif feature.startswith("C_"):
                family = "Component_C"
            elif feature.startswith("D_"):
                family = "Component_D"
            elif feature.startswith("mix_"):
                family = "Weighted_mixture"
            elif feature.startswith("avg_"):
                family = "Arithmetic_mean"
            elif feature.startswith("AB_"):
                family = "AB_interactions"
            elif feature in {"has_A", "has_B", "has_C", "has_D", "n_components", "val_total"}:
                family = "Composition"
            else:
                raise ValueError(
                    f"Cannot assign a static family to descriptor {feature!r}. "
                    "Extend the explicit encoding-prefix map before running."
                )
            records.append({"feature": feature, "family": family})
        pd.DataFrame(records).to_csv(path, index=False)
        print(f"Created deterministic descriptor-family map: {path}")
    meta = pd.read_csv(path)
    required = {"feature", "family"}
    if required - set(meta.columns):
        raise ValueError(f"{path.name} must contain {sorted(required)}.")
    meta = meta.loc[:, ["feature", "family"]].drop_duplicates("feature", keep="first")
    meta["feature"] = meta["feature"].astype(str)
    missing = sorted(set(all_features) - set(meta["feature"]))
    if missing:
        raise ValueError(
            "Feature metadata does not cover the full raw descriptor matrix; "
            f"missing {len(missing)} features, e.g. {missing[:10]}."
        )
    return meta.set_index("feature")


def diversity_terms(features: Iterable[str], candidates: Iterable[str], metadata: pd.DataFrame, orientation_by_feature: dict[str, str]) -> tuple[float, float]:
    """Exact diversity calculation used by the main DAGA pipeline."""
    selected_meta = metadata.loc[list(features)]
    candidate_meta = metadata.loc[list(candidates)]
    selected = selected_meta["family"].astype(str).to_numpy()
    all_families = sorted(candidate_meta["family"].astype(str).unique().tolist())
    counts = np.asarray([(selected == family).sum() for family in all_families], dtype=float)
    positive_counts = counts[counts > 0]
    proportions = positive_counts / positive_counts.sum()
    entropy = -(proportions * np.log(proportions)).sum()
    entropy /= np.log(len(positive_counts)) if len(positive_counts) > 1 else 1.0
    coverage = float((counts > 0).sum() / len(all_families)) if all_families else 0.0
    concentration = counts.max() / counts.sum()
    orientations = np.asarray([orientation_by_feature[f] for f in features], dtype=object)
    n_nades = int((orientations == "NADES_oriented").sum())
    n_non_nades = int((orientations == "non_NADES_oriented").sum())
    n_directional = n_nades + n_non_nades
    orientation = (1.0 - 2.0 * abs(n_nades / n_directional - 0.5)) if n_directional else 0.0
    diversity = (
        DIVERSITY_WEIGHTS["coverage"] * coverage
        + DIVERSITY_WEIGHTS["entropy"] * entropy
        + DIVERSITY_WEIGHTS["orientation"] * orientation
    )
    concentration_penalty = max(0.0, concentration - MAX_FAMILY_SHARE) / (1.0 - MAX_FAMILY_SHARE)
    return float(diversity), float(concentration_penalty)


def inner_oof_probabilities(X: pd.DataFrame, y: np.ndarray, cv_splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """OOF predictions on the fixed inner splits of one outer LOCO fold."""
    oof = np.full(len(y), np.nan, dtype=float)
    for fold_number, (tr, va) in enumerate(cv_splits, start=1):
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X.iloc[tr]).astype(np.float32)
        X_va = imputer.transform(X.iloc[va]).astype(np.float32)
        # Same model-seed construction as the main subset-60 DAGA pipeline.
        model = build_model(y[tr], MODEL_SEED + FINAL_SUBSET_SIZE * 100 + fold_number)
        model.fit(X_tr, y[tr])
        oof[va] = model.predict_proba(X_va)[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("Incomplete inner OOF predictions.")
    return oof


def find_balanced_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float, float]:
    """Main-pipeline threshold scan: returns tau, BAcc, MCC*, objective."""
    best: tuple[float, float, float, float] | None = None
    for threshold in THRESHOLD_GRID:
        pred = (probabilities >= threshold).astype(np.int8)
        bacc = float(balanced_accuracy_score(y, pred))
        mcc01 = float((matthews_corrcoef(y, pred) + 1.0) / 2.0)
        objective = 0.60 * bacc + 0.40 * mcc01
        if best is None or objective > best[3]:
            best = (float(threshold), bacc, mcc01, objective)
    if best is None:
        raise RuntimeError("Threshold optimization failed.")
    return best


def fitness(features: list[str], X: pd.DataFrame, y: np.ndarray, candidates: list[str], metadata: pd.DataFrame, orientation_by_feature: dict[str, str], cv_splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    probs = inner_oof_probabilities(X.loc[:, features], y, cv_splits)
    auc = roc_auc_score(y, probs)
    ap_non_nades = average_precision_score(1 - y, 1 - probs)
    _, bacc, mcc01, _ = find_balanced_threshold(y, probs)
    predictive = (
        PREDICTIVE_WEIGHTS["bacc"] * bacc
        + PREDICTIVE_WEIGHTS["mcc"] * mcc01
        + PREDICTIVE_WEIGHTS["ap_non_nades"] * ap_non_nades
        + PREDICTIVE_WEIGHTS["roc_auc"] * auc
    )
    diversity, concentration_penalty = diversity_terms(features, candidates, metadata, orientation_by_feature)
    return float(
        predictive + FITNESS_DIVERSITY_WEIGHT * diversity
        - CONCENTRATION_PENALTY_WEIGHT * concentration_penalty
    )


def tournament(population: list[list[int]], scores: list[float], rng: np.random.Generator) -> list[int]:
    selected = rng.choice(len(population), size=GA_TOURNAMENT_SIZE, replace=False)
    winner = max(selected, key=lambda i: scores[i])
    return population[int(winner)].copy()


def mutate(chromosome: list[int], n_candidates: int, rng: np.random.Generator) -> list[int]:
    result = chromosome.copy()
    if rng.random() >= GA_MUTATION_PROBABILITY:
        return result
    n_changes = int(rng.integers(1, 4))
    available = list(set(range(n_candidates)) - set(result))
    for position in rng.choice(len(result), size=min(n_changes, len(result)), replace=False):
        if not available:
            break
        replacement = int(rng.choice(available))
        available.remove(replacement)
        available.append(result[int(position)])
        result[int(position)] = replacement
    return sorted(result)


def crossover(a: list[int], b: list[int], n_candidates: int, rng: np.random.Generator) -> list[int]:
    if rng.random() >= GA_CROSSOVER_PROBABILITY:
        return a.copy()
    union = list(set(a) | set(b))
    chosen = list(rng.choice(union, size=min(FINAL_SUBSET_SIZE, len(union)), replace=False))
    if len(chosen) < FINAL_SUBSET_SIZE:
        remaining = list(set(range(n_candidates)) - set(chosen))
        chosen.extend(rng.choice(remaining, size=FINAL_SUBSET_SIZE - len(chosen), replace=False))
    return sorted(map(int, chosen))


def run_ga(X: pd.DataFrame, y: np.ndarray, candidates: list[str], metadata: pd.DataFrame, orientation_by_feature: dict[str, str], cv_splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[str], float]:
    """Run independent DAGA searches and return the highest-fitness subset."""
    best_features: list[str] | None = None
    best_score = -np.inf
    n_candidates = len(candidates)

    for run_number in range(1, N_GA_RUNS + 1):
        # Exact main-pipeline run seed for subset size 60.
        seed = MODEL_SEED + FINAL_SUBSET_SIZE * 10000 + 1000 * run_number
        rng = np.random.default_rng(seed)
        population = [
            sorted(rng.choice(n_candidates, size=FINAL_SUBSET_SIZE, replace=False).tolist())
            for _ in range(GA_POPULATION)
        ]
        stall = 0
        run_best = -np.inf
        for generation in range(GA_MAX_GENERATIONS):
            scores = [
                fitness([candidates[i] for i in chrom], X, y, candidates, metadata, orientation_by_feature, cv_splits)
                for chrom in population
            ]
            ranking = np.argsort(scores)[::-1]
            if scores[int(ranking[0])] > run_best + GA_MIN_IMPROVEMENT:
                run_best = scores[int(ranking[0])]
                stall = 0
            else:
                stall += 1
            if run_best > best_score:
                best_score = run_best
                best_features = [candidates[i] for i in population[int(ranking[0])]]
            if stall >= GA_STALL_GENERATIONS:
                break

            next_population = [population[int(i)].copy() for i in ranking[:GA_ELITE]]
            while len(next_population) < GA_POPULATION:
                parent_a = tournament(population, scores, rng)
                parent_b = tournament(population, scores, rng)
                child = mutate(crossover(parent_a, parent_b, n_candidates, rng), n_candidates, rng)
                next_population.append(child)
            population = next_population

    if best_features is None:
        raise RuntimeError("DAGA failed to select a descriptor subset.")
    return best_features, float(best_score)


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    return find_balanced_threshold(y, probabilities)[0]


def select_features_nested(X_outer_train: pd.DataFrame, y_outer_train: np.ndarray, metadata: pd.DataFrame, cv_splits: list[tuple[np.ndarray, np.ndarray]]) -> SelectionResult:
    X_qc = technical_qc(X_outer_train)
    X_imputed, ranked = supervised_ranking(X_qc, y_outer_train)
    candidates = correlation_prune(X_imputed, ranked)
    orientation_by_feature = ranked.set_index("feature").loc[candidates, "orientation"].to_dict()
    features, best_fitness = run_ga(X_qc.loc[:, candidates], y_outer_train, candidates, metadata, orientation_by_feature, cv_splits)
    return SelectionResult(
        features=features,
        fitness=best_fitness,
        n_after_qc=X_qc.shape[1],
        n_eligible=len(ranked),
        n_candidate_pool=len(candidates),
    )


def metric_row(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)) if both_classes else np.nan,
        "AP_NADES": float(average_precision_score(y_true, probabilities)) if both_classes else np.nan,
        "AP_non_NADES": float(average_precision_score(1 - y_true, 1 - probabilities)) if both_classes else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if both_classes else np.nan,
        "MCC": float(matthews_corrcoef(y_true, pred)) if both_classes else np.nan,
        "precision_NADES": float(precision_score(y_true, pred, zero_division=0)),
        "recall_NADES": float(recall_score(y_true, pred, zero_division=0)),
        "specificity_non_NADES": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# =============================================================================
# 3. LOAD DATA
# =============================================================================

master = pd.read_csv(MASTER_PATH, na_values=NA_VALUES, keep_default_na=True, low_memory=False)
original = pd.read_csv(ORIGINAL_PATH, low_memory=False)
required_master = {"Code", "activity"}
required_original = {"Code", *COMPONENT_COLUMNS}
if required_master - set(master):
    raise ValueError(f"master missing columns: {sorted(required_master - set(master))}")
if required_original - set(original):
    raise ValueError(f"original missing columns: {sorted(required_original - set(original))}")
if master["Code"].duplicated().any() or original["Code"].duplicated().any():
    raise ValueError("Code must be unique in both input files.")

df = master.merge(original[["Code", *COMPONENT_COLUMNS]], on="Code", how="inner", validate="one_to_one")
if len(df) != len(master):
    raise ValueError("Some master rows could not be matched to component identities.")
y = pd.to_numeric(df["activity"], errors="raise").astype(np.int8).to_numpy()
if set(np.unique(y)) - {0, 1}:
    raise ValueError("activity must be binary: 0 = non-NADES, 1 = NADES.")

metadata = {"Code", "activity", *COMPONENT_COLUMNS}
feature_columns = [c for c in master.columns if c not in metadata]
X_full = df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
feature_metadata = load_feature_metadata(FEATURE_METADATA_PATH, feature_columns)
component_frame = df.loc[:, COMPONENT_COLUMNS].astype("string").apply(lambda col: col.str.strip())
all_components = sorted(str(v) for v in pd.unique(pd.concat([component_frame[c] for c in COMPONENT_COLUMNS], ignore_index=True).dropna()) if str(v))

print(f"Samples: {len(df)} | raw descriptors: {X_full.shape[1]} | components: {len(all_components)}")


# =============================================================================
# 4. OUTER NESTED LOCO
# =============================================================================

fold_rows: list[dict] = []
prediction_tables: list[pd.DataFrame] = []
feature_tables: list[pd.DataFrame] = []

for number, component in enumerate(all_components, start=1):
    if OUTER_COMPONENTS is not None and component not in OUTER_COMPONENTS:
        continue
    held_out = component_frame.eq(component).any(axis=1).to_numpy()
    test_idx = np.flatnonzero(held_out)
    train_idx = np.flatnonzero(~held_out)
    y_train, y_test = y[train_idx], y[test_idx]
    if len(test_idx) < MIN_OUTER_TEST_SAMPLES:
        continue
    if REQUIRE_BOTH_OUTER_CLASSES and len(np.unique(y_test)) < 2:
        continue
    if len(np.unique(y_train)) < 2:
        continue

    print(f"\n[{number}/{len(all_components)}] Nested LOCO: {component} | train={len(train_idx)} | test={len(test_idx)}")
    X_train_raw = X_full.iloc[train_idx].copy()
    X_test_raw = X_full.iloc[test_idx].copy()

    # Exactly one fixed inner-CV partition is created for this outer fold and
    # reused by every chromosome, generation, GA run, and final threshold scan.
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    outer_cv_splits = list(inner_cv.split(X_train_raw, y_train))

    # The complete selection pipeline sees only X_train_raw and y_train.
    selection = select_features_nested(X_train_raw, y_train, feature_metadata, outer_cv_splits)
    selected = selection.features

    # The final operating point is selected from nested inner OOF predictions.
    inner_probs = inner_oof_probabilities(X_train_raw.loc[:, selected], y_train, outer_cv_splits)
    threshold = choose_threshold(y_train, inner_probs)

    # Train the outer-fold deployment model and predict the untouched component.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw.loc[:, selected]).astype(np.float32)
    X_test = imputer.transform(X_test_raw.loc[:, selected]).astype(np.float32)
    model = build_model(y_train, MODEL_SEED + number)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = metric_row(y_test, probabilities, threshold)

    fold_rows.append({
        "held_out_component": component,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "n_train_non_NADES": int((y_train == 0).sum()), "n_train_NADES": int((y_train == 1).sum()),
        "n_test_non_NADES": int((y_test == 0).sum()), "n_test_NADES": int((y_test == 1).sum()),
        "threshold_nested_oof": threshold,
        "inner_oof_fitness": selection.fitness,
        "n_descriptors_after_qc": selection.n_after_qc,
        "n_eligible_descriptors": selection.n_eligible,
        "n_candidate_pool": selection.n_candidate_pool,
        "n_final_features": len(selected),
        **metrics,
    })
    feature_tables.append(pd.DataFrame({"held_out_component": component, "rank_in_selected_set": np.arange(1, len(selected) + 1), "feature": selected}))
    prediction_tables.append(pd.DataFrame({
        "Code": df.iloc[test_idx]["Code"].astype(str).to_numpy(),
        "held_out_component": component,
        "true_activity": y_test,
        "probability_NADES": probabilities,
        "threshold_nested_oof": threshold,
        "predicted_activity": (probabilities >= threshold).astype(np.int8),
        "correct_prediction": (probabilities >= threshold).astype(np.int8) == y_test,
    }))
    print(f"  AUC={metrics['ROC_AUC']:.3f} | AP(non-NADES)={metrics['AP_non_NADES']:.3f} | BAcc={metrics['balanced_accuracy']:.3f} | MCC={metrics['MCC']:.3f} | tau={threshold:.2f}")


# =============================================================================
# 5. WRITE OUTPUTS
# =============================================================================

if not fold_rows:
    raise RuntimeError("No outer folds were evaluated. Check component names and eligibility settings.")

results = pd.DataFrame(fold_rows).sort_values("ROC_AUC", na_position="last").reset_index(drop=True)
selected_features = pd.concat(feature_tables, ignore_index=True)
predictions = pd.concat(prediction_tables, ignore_index=True)

results.to_csv(data_path('NestedLOCO_Final60/NestedLOCO_component_results.csv'), index=False)
selected_features.to_csv(data_path('NestedLOCO_Final60/NestedLOCO_fold_selected_features.csv'), index=False)
predictions.to_csv(data_path('NestedLOCO_Final60/NestedLOCO_sample_predictions.csv'), index=False)

summary_metrics = ["ROC_AUC", "AP_NADES", "AP_non_NADES", "balanced_accuracy", "MCC", "recall_NADES", "specificity_non_NADES"]
summary = pd.DataFrame([
    {"metric": metric, "n_valid_folds": results[metric].notna().sum(), "mean": results[metric].mean(), "median": results[metric].median(), "std": results[metric].std(ddof=1), "minimum": results[metric].min(), "maximum": results[metric].max()}
    for metric in summary_metrics
])
summary.to_csv(data_path('NestedLOCO_Final60/NestedLOCO_summary.csv'), index=False)

print("\nNested LOCO complete")
print(summary.to_string(index=False))
print(f"\nOutputs written to: {OUTPUT_DIR}")

