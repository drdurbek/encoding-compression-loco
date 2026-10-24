
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
# FINAL60 — TRAIN-ONLY REPRESENTATION / COMPRESSION COMPARISON
# ============================================================
#
# Compares three fixed descriptor spaces using only the 511-sample
# Train partition defined before supervised selection:
#
#   1. Full cleaned descriptor space
#   2. Train-only GA candidate pool (1000 descriptors)
#   3. Fixed Final60 descriptor representation
#
# The 128 internal-evaluation rows are never loaded into the analysis.
# Train only. 
#
# UMAP is a visualization. Local geometry is additionally quantified by
# k-nearest-neighbour retention relative to the full cleaned space.
# ============================================================

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

BASE = data_path('')

TRAIN_PATH = data_path('master_train_features.csv')

SPLIT_PATH = data_path('Final_clean_split_seed42.csv')

# Train-only candidate pool produced by the clean prefilter.
CANDIDATE_PATH = data_path('GA_candidate_descriptors_1000.csv')

# Fixed 60-descriptor representation selected from the clean DAGA sweep.
SELECTED_PATH = (
    data_path('Final60_Internal_Evaluation/Final60_selected_descriptors.csv')
)

OUTPUT_DIR = data_path('Final60_UMAP_Compression_Comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COORDINATES_OUT = (
    data_path('Final60_UMAP_Compression_Comparison/Final60_UMAP_coordinates_all_representations.csv')
)
SUMMARY_OUT = (
    data_path('Final60_UMAP_Compression_Comparison/Final60_UMAP_compression_summary.csv')
)
NEIGHBOR_OUT = (
    data_path('Final60_UMAP_Compression_Comparison/Final60_kNN_retention_per_sample.csv')
)
FIGURE_PNG = (
    data_path('Final60_UMAP_Compression_Comparison/Final60_UMAP_full_1000_60_comparison.png')
)
FIGURE_PDF = (
    data_path('Final60_UMAP_Compression_Comparison/Final60_UMAP_full_1000_60_comparison.pdf')
)

TARGET_COLUMN = "activity"
CODE_COLUMN = "Code"
SPLIT_TRAIN_LABEL = "Train"

SEED = 42
EXPECTED_TRAIN_SAMPLES = 511
EXPECTED_FINAL60_FEATURES = 60

MAX_MISSING_FRACTION = 0.95
N_PCA_COMPONENTS = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.10
K_NEIGHBORS = 10

NA_VALUES = [
    "na", "NA", "Na", "N/A", "n/a", "None", "none",
    "-", "", "inf", "-inf", "Inf", "+inf",
]

COLOR_NADES = "#91D1C2"
COLOR_NON_NADES = "#F39B7F"

plt.rcParams["font.family"] = "Arial"


# ============================================================
# 2. HELPERS
# ============================================================

def find_feature_column(frame, file_label):
    candidates = [
        "feature",
        "descriptor",
        "descriptor_name",
        "selected_feature",
    ]

    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    raise ValueError(
        f"No descriptor-name column was found in {file_label}.\n"
        f"Available columns: {frame.columns.tolist()}"
    )


def numeric_descriptor_frame(frame, features):
    return (
        frame
        .loc[:, features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def clean_full_space(X_train):
    """Apply unsupervised technical descriptor cleaning to Train only."""
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

    duplicate_mask = X_train.T.duplicated(keep="first")
    duplicates = X_train.columns[duplicate_mask].tolist()
    X_train = X_train.loc[:, ~duplicate_mask].copy()

    cleaning = pd.DataFrame({
        "stage": [
            "Initial train descriptor space",
            "All-NaN removed",
            ">95% missing removed",
            "Constant removed",
            "Exact duplicates removed",
            "Final cleaned descriptor space",
        ],
        "n_descriptors": [
            n_initial, len(all_nan), len(high_missing),
            len(constant), len(duplicates), X_train.shape[1],
        ],
    })
    return X_train, cleaning


def fit_representation(X_train, name):
    """Fit imputer, scaler, PCA, and UMAP on Train."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    n_pca = min(
        N_PCA_COMPONENTS,
        X_train_scaled.shape[0] - 1,
        X_train_scaled.shape[1],
    )

    if n_pca < 2:
        raise ValueError(
            f"{name}: insufficient dimensions for PCA/UMAP."
        )

    pca = PCA(
        n_components=n_pca,
        svd_solver="randomized",
        random_state=SEED,
    )

    X_train_pca = pca.fit_transform(X_train_scaled)
    

    umap_model = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
    )

    train_umap = umap_model.fit_transform(X_train_pca)
    

    return {
        "name": name,
        "train_umap": train_umap,
        "train_pca": X_train_pca,
        "n_descriptors": X_train.shape[1],
        "n_pca_components": n_pca,
        "pca_explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
    }


def coordinate_table(result, train_codes, train_y):
    train_coordinates = pd.DataFrame(
        {
            "Code": train_codes.astype(str).to_numpy(),
            "UMAP1": result["train_umap"][:, 0],
            "UMAP2": result["train_umap"][:, 1],
            "set": "train",
            "activity": train_y,
            "representation": result["name"],
        }
    )

    
    return pd.concat(
        [train_coordinates],
        ignore_index=True,
    )


def neighbor_retention(reference_pca, comparison_pca, comparison_name):
    """Fraction of a sample's full-space kNN retained after compression."""
    n_neighbors = min(K_NEIGHBORS + 1, len(reference_pca))

    reference_nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    ).fit(reference_pca)

    comparison_nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
    ).fit(comparison_pca)

    reference_indices = reference_nn.kneighbors(
        reference_pca,
        return_distance=False,
    )[:, 1:]

    comparison_indices = comparison_nn.kneighbors(
        comparison_pca,
        return_distance=False,
    )[:, 1:]

    effective_k = reference_indices.shape[1]
    retention = np.array(
        [
            len(set(ref) & set(comp)) / effective_k
            for ref, comp in zip(
                reference_indices,
                comparison_indices,
            )
        ],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "sample_index": np.arange(len(retention)),
            "representation": comparison_name,
            "k_neighbors": effective_k,
            "kNN_retention": retention,
        }
    )


def plot_three_panel(coordinate_sets, output_path):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6.3),
    )

    panel_labels = ["a", "b", "c"]

    for ax, coordinates, panel_label in zip(
        axes,
        coordinate_sets,
        panel_labels,
    ):
        nades = coordinates.loc[
            (coordinates["set"] == "train")
            & (coordinates["activity"] == 1)
        ]
        non_nades = coordinates.loc[
            (coordinates["set"] == "train")
            & (coordinates["activity"] == 0)
        ]
        
        ax.scatter(
            nades["UMAP1"],
            nades["UMAP2"],
            s=65,
            color=COLOR_NADES,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.85,
            label="NADES",
            zorder=2,
        )
        ax.scatter(
            non_nades["UMAP1"],
            non_nades["UMAP2"],
            s=65,
            color=COLOR_NON_NADES,
            edgecolors="white",
            linewidths=0.25,
            alpha=0.90,
            label="non-NADES",
            zorder=3,
        )
        
        ax.set_xlabel("UMAP1", fontsize=16)
        ax.set_ylabel("UMAP2", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        ax.set_title(
            coordinates["representation"].iloc[0],
            fontsize=14,
            pad=9,
        )
        ax.text(
            -0.10,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# 3. LOAD DATA AND FIXED CLEAN TRAIN PARTITION
# ============================================================

for path, description in [
    (TRAIN_PATH, "Labelled feature file"),
    (SPLIT_PATH, "Fixed clean split"),
    (CANDIDATE_PATH, "Train-only candidate-descriptor file"),
    (SELECTED_PATH, "Final60 selected-descriptor file"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")

labelled = pd.read_csv(
    TRAIN_PATH,
    na_values=NA_VALUES,
    keep_default_na=True,
    low_memory=False,
)

if not {CODE_COLUMN, TARGET_COLUMN}.issubset(labelled.columns):
    raise ValueError("Labelled data must contain Code and activity columns.")


y_all = pd.to_numeric(labelled[TARGET_COLUMN], errors="coerce")
valid_label = y_all.isin([0, 1])
labelled = labelled.loc[valid_label].copy()
y_all = y_all.loc[valid_label].astype(np.int8)

labelled[CODE_COLUMN] = labelled[CODE_COLUMN].astype("string").str.strip()

split_table = pd.read_csv(SPLIT_PATH, low_memory=False)
if not {CODE_COLUMN, "split"}.issubset(split_table.columns):
    raise ValueError("Final_clean_split_seed42.csv needs Code and split columns.")

split_table[CODE_COLUMN] = split_table[CODE_COLUMN].astype("string").str.strip()
split_map = dict(zip(split_table[CODE_COLUMN], split_table["split"]))
split_labels = labelled[CODE_COLUMN].map(split_map)

if split_labels.isna().any():
    raise ValueError("Some labelled Codes are absent from the fixed split file.")

train_mask = (split_labels == SPLIT_TRAIN_LABEL).to_numpy()
if int(train_mask.sum()) != EXPECTED_TRAIN_SAMPLES:
    raise ValueError(
        f"Expected {EXPECTED_TRAIN_SAMPLES} Train rows, found "
        f"{int(train_mask.sum())}."
    )

base_features = [
    column for column in labelled.columns
    if column not in {CODE_COLUMN, TARGET_COLUMN}
]



X_all_raw = numeric_descriptor_frame(labelled, base_features)


X_train_raw = X_all_raw.loc[train_mask].reset_index(drop=True)
y_train = y_all.loc[train_mask].to_numpy(dtype=np.int8)
codes_train = labelled.loc[train_mask, CODE_COLUMN].reset_index(drop=True)


# ============================================================
# 4. DEFINE THE THREE FROZEN REPRESENTATIONS
# ============================================================

X_train_clean, cleaning_summary = clean_full_space(
    X_train_raw,
    
)
cleaning_summary.to_csv(
    data_path('Final60_UMAP_Compression_Comparison/Final60_train_only_cleaning_summary.csv'),
    index=False,
)

candidate_table = pd.read_csv(CANDIDATE_PATH, low_memory=False)
selected_table = pd.read_csv(SELECTED_PATH, low_memory=False)

candidate_feature_col = find_feature_column(
    candidate_table,
    CANDIDATE_PATH.name,
)
selected_feature_col = find_feature_column(
    selected_table,
    SELECTED_PATH.name,
)

candidate_features = (
    candidate_table[candidate_feature_col]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)
selected_features = (
    selected_table[selected_feature_col]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

if len(candidate_features) != 1000:
    raise ValueError(
        f"Expected 1000 train-only candidate descriptors, found "
        f"{len(candidate_features)}."
    )
if len(selected_features) != EXPECTED_FINAL60_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FINAL60_FEATURES} Final60 descriptors, found "
        f"{len(selected_features)}."
    )

for feature_set, name in [
    (candidate_features, "GA candidate"),
    (selected_features, "Final60"),
]:
    missing = [
        feature for feature in feature_set
        if feature not in X_train_clean.columns
    ]
    if missing:
        raise ValueError(
            f"{name} descriptors missing after train-only cleaning:\n"
            + "\n".join(missing[:20])
        )

X_train_1000 = X_train_clean.loc[:, candidate_features].copy()


X_train_60 = X_train_clean.loc[:, selected_features].copy()



# ============================================================
# 5. TRAIN-ONLY UMAP EMBEDDINGS
# ============================================================

print("=" * 94)
print("FINAL60 TRAIN-ONLY REPRESENTATION COMPARISON")
print("=" * 94)
print(f"Train samples              : {len(y_train)}")
print(
    "Train NADES / non-NADES   : "
    f"{int((y_train == 1).sum())} / {int((y_train == 0).sum())}"
)
print("Internal evaluation rows   : excluded")
print(cleaning_summary.to_string(index=False))

full_result = fit_representation(
    X_train_clean,

    "Full cleaned space",
)
candidate_result = fit_representation(
    X_train_1000,

    "Train-only GA candidate pool (1000)",
)
final60_result = fit_representation(
    X_train_60,

    "Fixed Final60 representation",
)

full_coordinates = coordinate_table(
    full_result,
    codes_train,
    y_train,

)
candidate_coordinates = coordinate_table(
    candidate_result,
    codes_train,
    y_train,

)
final60_coordinates = coordinate_table(
    final60_result,
    codes_train,
    y_train,

)

all_coordinates = pd.concat(
    [full_coordinates, candidate_coordinates, final60_coordinates],
    ignore_index=True,
)
all_coordinates.to_csv(COORDINATES_OUT, index=False)


# ============================================================
# 6. QUANTIFY LOCAL GEOMETRY RETENTION
# ============================================================

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

retention = pd.concat(
    [candidate_retention, final60_retention],
    ignore_index=True,
)
retention["Code"] = np.tile(
    codes_train.astype(str).to_numpy(),
    2,
)
retention["activity"] = np.tile(y_train, 2)
retention.to_csv(NEIGHBOR_OUT, index=False)

summary = pd.DataFrame(
    [
        {
            "representation": full_result["name"],
            "n_descriptors": full_result["n_descriptors"],
            "n_pca_components": full_result["n_pca_components"],
            "pca_explained_variance": full_result[
                "pca_explained_variance"
            ],
            "mean_kNN_retention_vs_full": 1.0,
            "median_kNN_retention_vs_full": 1.0,
        },
        {
            "representation": candidate_result["name"],
            "n_descriptors": candidate_result["n_descriptors"],
            "n_pca_components": candidate_result["n_pca_components"],
            "pca_explained_variance": candidate_result[
                "pca_explained_variance"
            ],
            "mean_kNN_retention_vs_full": candidate_retention[
                "kNN_retention"
            ].mean(),
            "median_kNN_retention_vs_full": candidate_retention[
                "kNN_retention"
            ].median(),
        },
        {
            "representation": final60_result["name"],
            "n_descriptors": final60_result["n_descriptors"],
            "n_pca_components": final60_result["n_pca_components"],
            "pca_explained_variance": final60_result[
                "pca_explained_variance"
            ],
            "mean_kNN_retention_vs_full": final60_retention[
                "kNN_retention"
            ].mean(),
            "median_kNN_retention_vs_full": final60_retention[
                "kNN_retention"
            ].median(),
        },
    ]
)
summary["k_neighbors"] = K_NEIGHBORS
summary["UMAP_n_neighbors"] = UMAP_N_NEIGHBORS
summary["UMAP_min_dist"] = UMAP_MIN_DIST
summary.to_csv(SUMMARY_OUT, index=False)


# ============================================================
# 7. PLOT AND REPORT
# ============================================================

plot_three_panel(
    [
        full_coordinates,
        candidate_coordinates,
        final60_coordinates,
    ],
    FIGURE_PNG,
)

print("\n" + "=" * 94)
print("TRAIN-ONLY COMPRESSION / GEOMETRY SUMMARY")
print("=" * 94)
print(
    summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nSaved:")
for path in [
    COORDINATES_OUT,
    SUMMARY_OUT,
    NEIGHBOR_OUT,
    FIGURE_PNG,
    FIGURE_PDF,
]:
    print(" -", path)

print(
    "\nInterpretation: UMAP is visual evidence only. The kNN-retention "
    "values quantify how much local structure from the full cleaned "
    "representation remains in the 1000- and 60-descriptor spaces."
)

