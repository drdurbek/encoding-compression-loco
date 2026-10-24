
"""Train-only geometry comparison for the Tg full, candidate, and Final60 spaces.

This is a descriptive compression analysis.  UMAP is fitted only on the
208-row outer-training partition and is not used in descriptor, RF, or
hyperparameter selection.  The 55 held-out parent-NADES groups are excluded.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE = data_path('TG')

TRAIN_PATH = data_path('TG/master_train_features.csv')
SPLIT_PATH = data_path('TG/TG_clean_split_seed42.csv')
CANDIDATE_PATH = data_path('TG/TG_GA_candidate_descriptors_1000.csv')
SELECTED_PATH = (
    data_path('TG/TG_RF_DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('TG/TG_Final60_UMAP_Compression_Comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COORDINATES_OUT = data_path('TG/TG_Final60_UMAP_Compression_Comparison/TG_Final60_UMAP_coordinates_train_only.csv')
SUMMARY_OUT = data_path('TG/TG_Final60_UMAP_Compression_Comparison/TG_Final60_compression_geometry_summary.csv')
NEIGHBOR_OUT = data_path('TG/TG_Final60_UMAP_Compression_Comparison/TG_Final60_kNN_retention_per_sample.csv')
FIGURE_PNG = data_path('TG/TG_Final60_UMAP_Compression_Comparison/TG_Final60_UMAP_full_560_60_comparison.png')
FIGURE_PDF = data_path('TG/TG_Final60_UMAP_Compression_Comparison/TG_Final60_UMAP_full_560_60_comparison.pdf')

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"       # Legacy column; continuous Tg in K.
GROUP_COLUMN = "Core_NADES_ID"   # Metadata only; never a descriptor.

SEED = 42
EXPECTED_OUTER_TRAIN_ROWS = 208
EXPECTED_CANDIDATES = 560
EXPECTED_FINAL_FEATURES = 60
MAX_MISSING_FRACTION = 0.95
N_PCA_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.10
K_NEIGHBORS = 10

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

plt.rcParams["font.family"] = "Arial"


# ============================================================
# 2. HELPERS
# ============================================================

def normalise_code(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def find_feature_column(frame, file_label):
    accepted_names = [
        "feature",
        "descriptor",
        "descriptor_name",
        "selected_feature",
    ]
    lookup = {str(column).strip().lower(): column for column in frame.columns}

    for name in accepted_names:
        if name in lookup:
            return lookup[name]

    raise ValueError(
        f"No descriptor-name column was found in {file_label}.\n"
        f"Columns: {frame.columns.tolist()}"
    )


def technical_clean_train_only(X_train):
    """Train-only, unsupervised removal of unusable descriptor columns."""
    n_initial = X_train.shape[1]

    all_nan = X_train.columns[X_train.isna().all()].tolist()
    X_train = X_train.drop(columns=all_nan)

    high_missing = X_train.columns[
        X_train.isna().mean() > MAX_MISSING_FRACTION
    ].tolist()
    X_train = X_train.drop(columns=high_missing)

    constant = X_train.columns[
        X_train.nunique(dropna=True) <= 1
    ].tolist()
    X_train = X_train.drop(columns=constant)

    duplicates = X_train.columns[X_train.T.duplicated(keep="first")].tolist()
    X_train = X_train.loc[:, ~X_train.T.duplicated(keep="first")].copy()

    summary = pd.DataFrame({
        "stage": [
            "Initial descriptor space",
            "All-NaN removed",
            ">95% missing removed",
            "Constant removed",
            "Exact duplicates removed",
            "Final train-only cleaned space",
        ],
        "n_descriptors": [
            n_initial,
            len(all_nan),
            len(high_missing),
            len(constant),
            len(duplicates),
            X_train.shape[1],
        ],
    })
    return X_train, summary


def fit_representation(X_train, name):
    """Fit preprocessing, PCA, and UMAP on one frozen training representation."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)

    n_pca = min(
        N_PCA_COMPONENTS,
        X_scaled.shape[0] - 1,
        X_scaled.shape[1],
    )
    if n_pca < 2:
        raise ValueError(f"{name}: too few usable dimensions for PCA/UMAP.")

    pca = PCA(
        n_components=n_pca,
        svd_solver="randomized",
        random_state=SEED,
    )
    X_pca = pca.fit_transform(X_scaled)

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
    )
    X_umap = umap_model.fit_transform(X_pca)

    return {
        "name": name,
        "train_pca": X_pca,
        "train_umap": X_umap,
        "n_descriptors": X_train.shape[1],
        "n_pca_components": n_pca,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def neighbor_retention(reference_space, compressed_space, representation_name):
    """Fraction of 10 nearest neighbours retained after compression."""
    n_neighbors = min(K_NEIGHBORS + 1, len(reference_space))
    ref_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(reference_space)
    cmp_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(compressed_space)

    ref_indices = ref_neighbors.kneighbors(return_distance=False)[:, 1:]
    cmp_indices = cmp_neighbors.kneighbors(return_distance=False)[:, 1:]

    retention = [
        len(set(ref_row).intersection(set(cmp_row))) / K_NEIGHBORS
        for ref_row, cmp_row in zip(ref_indices, cmp_indices)
    ]

    return pd.DataFrame({
        "representation": representation_name,
        "kNN_retention": retention,
    })


def coordinate_table(result, codes, groups, tg_values):
    return pd.DataFrame({
        "representation": result["name"],
        CODE_COLUMN: codes.astype(str).to_numpy(),
        GROUP_COLUMN: groups.astype(str).to_numpy(),
        "Tg_K": tg_values,
        "UMAP1": result["train_umap"][:, 0],
        "UMAP2": result["train_umap"][:, 1],
    })


def plot_three_panels(coordinate_tables, out_png, out_pdf):
    all_tg = np.concatenate([table["Tg_K"].to_numpy() for table in coordinate_tables])
    norm = plt.Normalize(vmin=all_tg.min(), vmax=all_tg.max())
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))

    for label, ax, table in zip(["d", "e", "f"], axes, coordinate_tables):
        scatter = ax.scatter(
            table["UMAP1"],
            table["UMAP2"],
            c=table["Tg_K"],
            cmap=cmap,
            norm=norm,
            s=48,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.set_title(table["representation"].iloc[0], fontsize=14)
        ax.set_xlabel("UMAP1", fontsize=14)
        ax.set_ylabel("UMAP2", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.text(
            -0.10, 1.03, label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
        )

    colorbar = fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.045, aspect=45, pad=0.02, shrink=0.93)
    colorbar.set_label("Glass-transition temperature, Tg (K)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)

    fig.tight_layout(rect=[0, 0.18, 1, 1])  # поднимает панели, создаёт место снизу

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# 3. LOAD THE FROZEN OUTER-TRAINING PARTITION
# ============================================================

for path, description in [
    (TRAIN_PATH, "Tg master feature table"),
    (SPLIT_PATH, "frozen group split"),
    (CANDIDATE_PATH, "train-only GA candidate table"),
    (SELECTED_PATH, "Final60 descriptor table"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")

data = pd.read_csv(TRAIN_PATH, na_values=NA_VALUES, keep_default_na=True, low_memory=False)
split_ref = pd.read_csv(SPLIT_PATH, low_memory=False)
candidate_table = pd.read_csv(CANDIDATE_PATH, low_memory=False)
selected_table = pd.read_csv(SELECTED_PATH, low_memory=False)

required_data = {CODE_COLUMN, TARGET_COLUMN, GROUP_COLUMN}
if not required_data.issubset(data.columns):
    raise ValueError(f"Dataset must contain {sorted(required_data)}.")
if not {CODE_COLUMN, "split"}.issubset(split_ref.columns):
    raise ValueError("Split table must contain Code and split columns.")

data[CODE_COLUMN] = normalise_code(data[CODE_COLUMN])
split_ref[CODE_COLUMN] = normalise_code(split_ref[CODE_COLUMN])
data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
if data[TARGET_COLUMN].isna().any():
    raise ValueError("activity contains missing or non-numeric Tg values.")

split_map = dict(zip(split_ref[CODE_COLUMN], split_ref["split"]))
split_label = data[CODE_COLUMN].map(split_map)
if split_label.isna().any():
    raise ValueError("Some master-table codes are absent from the frozen split.")

train_mask = (
    split_label.astype("string").str.strip().str.lower()
    .isin({"train", "outer_train", "training"})
    .to_numpy()
)
if train_mask.sum() != EXPECTED_OUTER_TRAIN_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_OUTER_TRAIN_ROWS} outer-training rows; "
        f"found {train_mask.sum()}."
    )

base_features = [
    column for column in data.columns
    if column not in {CODE_COLUMN, TARGET_COLUMN, GROUP_COLUMN}
]
X_all_raw = (
    data[base_features]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
)

X_train_raw = X_all_raw.loc[train_mask].reset_index(drop=True)
codes_train = data.loc[train_mask, CODE_COLUMN].reset_index(drop=True)
groups_train = data.loc[train_mask, GROUP_COLUMN].reset_index(drop=True)
tg_train = data.loc[train_mask, TARGET_COLUMN].to_numpy(dtype=float)


# ============================================================
# 4. DEFINE FROZEN FULL, CANDIDATE, AND FINAL60 SPACES
# ============================================================

X_train_clean, cleaning_summary = technical_clean_train_only(X_train_raw)
cleaning_summary.to_csv(data_path('TG/TG_Final60_UMAP_Compression_Comparison/TG_Final60_train_only_cleaning_summary.csv'), index=False)

candidate_feature_column = find_feature_column(candidate_table, CANDIDATE_PATH.name)
selected_feature_column = find_feature_column(selected_table, SELECTED_PATH.name)

candidate_features = (
    candidate_table[candidate_feature_column]
    .dropna().astype(str).str.strip().drop_duplicates().tolist()
)
selected_features = (
    selected_table[selected_feature_column]
    .dropna().astype(str).str.strip().drop_duplicates().tolist()
)

if len(candidate_features) != EXPECTED_CANDIDATES:
    raise ValueError(
        f"Expected {EXPECTED_CANDIDATES} candidate descriptors; "
        f"found {len(candidate_features)}."
    )
if len(selected_features) != EXPECTED_FINAL_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FINAL_FEATURES} Final60 descriptors; "
        f"found {len(selected_features)}."
    )

for feature_set, name in [
    (candidate_features, "Candidate"),
    (selected_features, "Final60"),
]:
    missing = [feature for feature in feature_set if feature not in X_train_clean.columns]
    if missing:
        raise ValueError(
            f"{name} descriptors absent after train-only cleaning: {missing[:15]}"
        )

X_train_candidate = X_train_clean.loc[:, candidate_features].copy()
X_train_final60 = X_train_clean.loc[:, selected_features].copy()


# ============================================================
# 5. FIT TRAIN-ONLY EMBEDDINGS AND QUANTIFY GEOMETRY
# ============================================================

print("=" * 94)
print("TG FINAL60 TRAIN-ONLY COMPRESSION / GEOMETRY COMPARISON")
print("=" * 94)
print(f"Outer-train rows          : {len(tg_train)}")
print(f"Outer-train parent NADES  : {groups_train.nunique()}")
print("Held-out rows             : excluded")
print(cleaning_summary.to_string(index=False))

full_result = fit_representation(X_train_clean, "Full cleaned space")
candidate_result = fit_representation(X_train_candidate, "Train-only GA candidate pool (560)")
final60_result = fit_representation(X_train_final60, "Fixed Final60 representation")

full_coordinates = coordinate_table(full_result, codes_train, groups_train, tg_train)
candidate_coordinates = coordinate_table(candidate_result, codes_train, groups_train, tg_train)
final60_coordinates = coordinate_table(final60_result, codes_train, groups_train, tg_train)

pd.concat(
    [full_coordinates, candidate_coordinates, final60_coordinates],
    ignore_index=True,
).to_csv(COORDINATES_OUT, index=False)

candidate_retention = neighbor_retention(
    full_result["train_pca"],
    candidate_result["train_pca"],
    candidate_result["name"],
)
final60_retention = neighbor_retention(
    full_result["train_pca"],
    final60_result["train_pca"],
    final60_result["name"],
)

retention = pd.concat([candidate_retention, final60_retention], ignore_index=True)
retention[CODE_COLUMN] = np.tile(codes_train.astype(str).to_numpy(), 2)
retention[GROUP_COLUMN] = np.tile(groups_train.astype(str).to_numpy(), 2)
retention["Tg_K"] = np.tile(tg_train, 2)
retention.to_csv(NEIGHBOR_OUT, index=False)

summary = pd.DataFrame([
    {
        "representation": full_result["name"],
        "n_descriptors": full_result["n_descriptors"],
        "n_pca_components": full_result["n_pca_components"],
        "pca_explained_variance": full_result["pca_explained_variance"],
        "mean_kNN_retention_vs_full": 1.0,
        "median_kNN_retention_vs_full": 1.0,
    },
    {
        "representation": candidate_result["name"],
        "n_descriptors": candidate_result["n_descriptors"],
        "n_pca_components": candidate_result["n_pca_components"],
        "pca_explained_variance": candidate_result["pca_explained_variance"],
        "mean_kNN_retention_vs_full": candidate_retention["kNN_retention"].mean(),
        "median_kNN_retention_vs_full": candidate_retention["kNN_retention"].median(),
    },
    {
        "representation": final60_result["name"],
        "n_descriptors": final60_result["n_descriptors"],
        "n_pca_components": final60_result["n_pca_components"],
        "pca_explained_variance": final60_result["pca_explained_variance"],
        "mean_kNN_retention_vs_full": final60_retention["kNN_retention"].mean(),
        "median_kNN_retention_vs_full": final60_retention["kNN_retention"].median(),
    },
])
summary["k_neighbors"] = K_NEIGHBORS
summary["UMAP_n_neighbors"] = UMAP_N_NEIGHBORS
summary["UMAP_min_dist"] = UMAP_MIN_DIST
summary.to_csv(SUMMARY_OUT, index=False)

plot_three_panels(
    [full_coordinates, candidate_coordinates, final60_coordinates],
    FIGURE_PNG,
    FIGURE_PDF,
)

print("\n" + "=" * 94)
print("TG FINAL60 COMPRESSION / GEOMETRY SUMMARY")
print("=" * 94)
print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nSaved:")
for path in [COORDINATES_OUT, SUMMARY_OUT, NEIGHBOR_OUT, FIGURE_PNG, FIGURE_PDF]:
    print(" -", path)
print("\nUMAP is descriptive only; kNN retention quantifies local geometry preservation.")

