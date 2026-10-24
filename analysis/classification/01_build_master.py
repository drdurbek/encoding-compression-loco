
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
# BUILD RAW STOICHIOMETRY-AWARE MASTER DATASETS
# A/B/C/D component tables -> train and blind master matrices
#
# Feature cleaning, imputation, scaling, PCA and model fitting
# are performed later in the modeling pipeline.
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 0. PATHS AND SETTINGS
# ============================================================

DATA = data_path('')

A_PATH = data_path('a.csv')
B_PATH = data_path('b.csv')
C_PATH = data_path('c.csv')
D_PATH = data_path('d.csv')

TRAIN_LABELS = data_path('original.csv')   # Code, activity
BLIND_CODES = data_path('blind.csv')       # Code

TRAIN_OUT = data_path('master_train_features.csv')
BLIND_OUT = data_path('master_blind_features.csv')

COMPONENTS = ("A", "B", "C", "D")

EPS = 1e-6

# "directed" reproduces:
#     A / (B + eps)
#
# "symmetric" produces a truly permutation-invariant ratio:
#     max(|A|, |B|) / (min(|A|, |B|) + eps)
#
# Use "directed" if the current article results were obtained
# using the original A/B formula.
PAIRWISE_RATIO_MODE = "directed"


# ============================================================
# 1. GENERAL UTILITIES
# ============================================================

def normalize_code(series: pd.Series) -> pd.Series:
    """
    Normalize mixture codes while preserving missing values.
    """
    result = series.astype("string").str.strip()

    # Removes an artificial '.0' if codes were imported as floats
    result = result.str.replace(r"\.0$", "", regex=True)

    return result


def check_file_exists(path: Path) -> None:
    """
    Check that a required input file exists.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def validate_unique_codes(
    df: pd.DataFrame,
    source_name: str,
) -> None:
    """
    Verify that each table contains one row per Code.
    """
    if "Code" not in df.columns:
        raise ValueError(f"{source_name}: column 'Code' is missing.")

    if df["Code"].isna().any():
        n_missing = int(df["Code"].isna().sum())

        raise ValueError(
            f"{source_name}: found {n_missing} missing Code values."
        )

    duplicated = df["Code"].duplicated(keep=False)

    if duplicated.any():
        examples = (
            df.loc[duplicated, "Code"]
            .astype(str)
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            f"{source_name}: Code must be unique. "
            f"Duplicated examples: {examples}"
        )


# ============================================================
# 2. LOAD COMPONENT TABLES
# ============================================================

def load_component(
    path: Path,
    prefix: str,
    val_col: str,
) -> pd.DataFrame:
    """
    Load one component table.

    Required columns:
        Code
        val1 / val2 / val3 / val4

    All molecular descriptors receive a component prefix:
        MW -> A_MW
        MW -> B_MW
        etc.
    """
    check_file_exists(path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {"Code", val_col}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name}: required columns are {required}; "
            f"available columns are {set(df.columns)}"
        )

    df = df.copy()

    df["Code"] = normalize_code(df["Code"])

    validate_unique_codes(
        df,
        path.name,
    )

    rename_map = {
        col: f"{prefix}_{col}"
        for col in df.columns
        if col not in {"Code", val_col}
    }

    out = df.rename(columns=rename_map)

    out = out.rename(
        columns={
            val_col: f"{prefix}_val"
        }
    )

    out[f"{prefix}_val"] = pd.to_numeric(
        out[f"{prefix}_val"],
        errors="coerce",
    )

    negative_mask = out[f"{prefix}_val"] < 0

    if negative_mask.any():
        examples = (
            out.loc[
                negative_mask,
                ["Code", f"{prefix}_val"],
            ]
            .head(20)
            .to_dict("records")
        )

        raise ValueError(
            f"{path.name}: negative stoichiometric coefficients "
            f"were found. Examples: {examples}"
        )

    # Convert molecular descriptors to numeric
    for col in out.columns:
        if col != "Code":
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    print(
        f"Loaded {path.name}: "
        f"{out.shape[0]} rows × {out.shape[1]} columns"
    )

    return out


# ============================================================
# 3. MERGE COMPONENT TABLES
# ============================================================

def merge_components(
    component_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Outer merge by Code.

    validate='one_to_one' prevents accidental Cartesian
    multiplication when duplicate codes exist.
    """
    if not component_tables:
        raise ValueError("No component tables were provided.")

    merged = component_tables[0].copy()

    for component_df in component_tables[1:]:
        merged = merged.merge(
            component_df,
            on="Code",
            how="outer",
            validate="one_to_one",
        )

    merged = (
        merged
        .sort_values("Code")
        .reset_index(drop=True)
    )

    return merged


# ============================================================
# 4. COMPONENT PRESENCE AND STOICHIOMETRY
# ============================================================

def add_stoichiometry_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add:

        A_val_filled ... D_val_filled
        has_A ... has_D
        n_components
        val_total
        A_val_frac ... D_val_frac
    """
    out = df.copy()

    for component in COMPONENTS:
        val_col = f"{component}_val"

        if val_col not in out.columns:
            out[val_col] = np.nan

        out[val_col] = pd.to_numeric(
            out[val_col],
            errors="coerce",
        )

        out[f"{component}_val_filled"] = (
            out[val_col]
            .fillna(0.0)
            .astype(float)
        )

        out[f"has_{component}"] = (
            out[f"{component}_val_filled"] > 0
        ).astype(np.int8)

    presence_cols = [
        f"has_{component}"
        for component in COMPONENTS
    ]

    out["n_components"] = (
        out[presence_cols]
        .sum(axis=1)
        .astype(np.int8)
    )

    filled_val_cols = [
        f"{component}_val_filled"
        for component in COMPONENTS
    ]

    out["val_total"] = (
        out[filled_val_cols]
        .sum(axis=1)
    )

    for component in COMPONENTS:
        out[f"{component}_val_frac"] = np.where(
            out["val_total"] > 0,
            out[f"{component}_val_filled"]
            / out["val_total"],
            np.nan,
        )

    invalid_total = out["val_total"] <= 0

    if invalid_total.any():
        examples = (
            out.loc[invalid_total, "Code"]
            .head(20)
            .tolist()
        )

        warnings.warn(
            f"{int(invalid_total.sum())} mixtures have val_total <= 0. "
            f"Examples: {examples}",
            stacklevel=2,
        )

    invalid_component_count = out["n_components"] < 2

    if invalid_component_count.any():
        examples = (
            out.loc[
                invalid_component_count,
                ["Code", "n_components"],
            ]
            .head(20)
            .to_dict("records")
        )

        warnings.warn(
            f"{int(invalid_component_count.sum())} mixtures contain "
            f"fewer than two components. Examples: {examples}",
            stacklevel=2,
        )

    return out


# ============================================================
# 5. IDENTIFY MOLECULAR DESCRIPTOR BASES
# ============================================================

def get_descriptor_bases(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return descriptor names represented in at least one
    component table.

    Example:
        A_MW, B_MW, C_MW -> descriptor base 'MW'
    """
    excluded = {
        "val",
        "val_filled",
        "val_frac",
    }

    bases = set()

    for component in COMPONENTS:
        prefix = f"{component}_"

        for col in df.columns:
            if not col.startswith(prefix):
                continue

            base = col[len(prefix):]

            if base in excluded:
                continue

            if base.startswith("val"):
                continue

            bases.add(base)

    return sorted(bases)


def get_shared_ab_bases(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return descriptors available for both A and B.
    """
    excluded = {
        "val",
        "val_filled",
        "val_frac",
    }

    bases = []

    for col in df.columns:
        if not col.startswith("A_"):
            continue

        base = col[2:]

        if base in excluded:
            continue

        if base.startswith("val"):
            continue

        if f"B_{base}" in df.columns:
            bases.append(base)

    return sorted(set(bases))


# ============================================================
# 6. ARITHMETIC MEAN DESCRIPTORS
# ============================================================

def add_arithmetic_mean_descriptors(
    df: pd.DataFrame,
    descriptor_bases: list[str],
) -> pd.DataFrame:
    """
    Calculate an unweighted arithmetic mean across all
    components present in a mixture:

                 sum(D_p for present components)
        avg_D = --------------------------------
                         n_components

    If a present component lacks the descriptor value,
    the corresponding mixture descriptor remains NaN.
    """
    out = df.copy()

    generated = {}

    print(
        f"\nGenerating arithmetic mean descriptors "
        f"for {len(descriptor_bases)} descriptor bases..."
    )

    for base in descriptor_bases:
        descriptor_sum = pd.Series(
            0.0,
            index=out.index,
            dtype=float,
        )

        valid = pd.Series(
            True,
            index=out.index,
            dtype=bool,
        )

        for component in COMPONENTS:
            present = (
                out[f"{component}_val_filled"] > 0
            )

            descriptor_col = f"{component}_{base}"

            if descriptor_col in out.columns:
                descriptor = pd.to_numeric(
                    out[descriptor_col],
                    errors="coerce",
                )

                valid &= ~(
                    present
                    & descriptor.isna()
                )

                descriptor_sum += (
                    descriptor
                    .fillna(0.0)
                    .where(present, 0.0)
                )

            else:
                valid &= ~present

        result = (
            descriptor_sum
            / out["n_components"].replace(0, np.nan)
        )

        result.loc[~valid] = np.nan

        generated[f"avg_{base}"] = result

    if generated:
        generated_df = pd.DataFrame(
            generated,
            index=out.index,
        )

        out = pd.concat(
            [out, generated_df],
            axis=1,
        )

    return out


# ============================================================
# 7. STOICHIOMETRY-WEIGHTED MIXTURE DESCRIPTORS
# ============================================================

def add_weighted_mixture_descriptors(
    df: pd.DataFrame,
    descriptor_bases: list[str],
) -> pd.DataFrame:
    """
    Calculate:

        mix_D =
            v_A D_A + v_B D_B + v_C D_C + v_D D_D
            ---------------------------------------
                         val_total

    Missing descriptors are tolerated only for absent components.
    """
    out = df.copy()

    generated = {}

    denominator = out["val_total"].replace(
        0,
        np.nan,
    )

    print(
        f"Generating stoichiometry-weighted descriptors "
        f"for {len(descriptor_bases)} descriptor bases..."
    )

    for base in descriptor_bases:
        numerator = pd.Series(
            0.0,
            index=out.index,
            dtype=float,
        )

        valid = pd.Series(
            True,
            index=out.index,
            dtype=bool,
        )

        for component in COMPONENTS:
            coefficient = out[
                f"{component}_val_filled"
            ]

            present = coefficient > 0

            descriptor_col = f"{component}_{base}"

            if descriptor_col in out.columns:
                descriptor = pd.to_numeric(
                    out[descriptor_col],
                    errors="coerce",
                )

                valid &= ~(
                    present
                    & descriptor.isna()
                )

                numerator += (
                    coefficient
                    * descriptor.fillna(0.0)
                )

            else:
                valid &= ~present

        result = numerator / denominator
        result.loc[~valid] = np.nan

        generated[f"mix_{base}"] = result

    if generated:
        generated_df = pd.DataFrame(
            generated,
            index=out.index,
        )

        out = pd.concat(
            [out, generated_df],
            axis=1,
        )

    return out


# ============================================================
# 8. A/B INTERACTION DESCRIPTORS
# ============================================================

def calculate_ab_ratio(
    a: pd.Series,
    b: pd.Series,
    mode: str,
    eps: float,
) -> pd.Series:
    """
    Calculate either a directed or symmetric ratio.
    """
    if mode == "directed":
        ratio = a / (b + eps)

    elif mode == "symmetric":
        abs_a = a.abs()
        abs_b = b.abs()

        maximum = pd.concat(
            [abs_a, abs_b],
            axis=1,
        ).max(axis=1)

        minimum = pd.concat(
            [abs_a, abs_b],
            axis=1,
        ).min(axis=1)

        ratio = maximum / (minimum + eps)

    else:
        raise ValueError(
            "PAIRWISE_RATIO_MODE must be "
            "'directed' or 'symmetric'."
        )

    ratio[a.isna() | b.isna()] = np.nan

    return ratio


def add_ab_interaction_descriptors(
    df: pd.DataFrame,
    shared_ab_bases: list[str],
    ratio_mode: str,
    eps: float,
) -> pd.DataFrame:
    """
    Generate:

        AB_avg_D
        AB_diff_D
        AB_prod_D
        AB_ratio_D
    """
    out = df.copy()

    generated = {}

    print(
        f"Generating A/B interaction descriptors "
        f"for {len(shared_ab_bases)} descriptor bases..."
    )

    print(
        f"A/B ratio mode: {ratio_mode}"
    )

    for base in shared_ab_bases:
        a = pd.to_numeric(
            out[f"A_{base}"],
            errors="coerce",
        )

        b = pd.to_numeric(
            out[f"B_{base}"],
            errors="coerce",
        )

        generated[f"AB_avg_{base}"] = (
            a + b
        ) / 2.0

        generated[f"AB_diff_{base}"] = (
            a - b
        ).abs()

        generated[f"AB_prod_{base}"] = (
            a * b
        )

        generated[f"AB_ratio_{base}"] = (
            calculate_ab_ratio(
                a=a,
                b=b,
                mode=ratio_mode,
                eps=eps,
            )
        )

    # Stoichiometric A/B interactions
    a_val = pd.to_numeric(
        out["A_val"],
        errors="coerce",
    )

    b_val = pd.to_numeric(
        out["B_val"],
        errors="coerce",
    )

    generated["AB_avg_val"] = (
        a_val + b_val
    ) / 2.0

    generated["AB_diff_val"] = (
        a_val - b_val
    ).abs()

    generated["AB_prod_val"] = (
        a_val * b_val
    )

    generated["AB_ratio_val"] = calculate_ab_ratio(
        a=a_val,
        b=b_val,
        mode=ratio_mode,
        eps=eps,
    )

    generated_df = pd.DataFrame(
        generated,
        index=out.index,
    )

    out = pd.concat(
        [out, generated_df],
        axis=1,
    )

    return out


# ============================================================
# 9. LOAD LABELS AND BLIND CODES
# ============================================================

def load_train_labels(
    path: Path,
) -> pd.DataFrame:
    """
    Load and validate Code/activity labels.
    """
    check_file_exists(path)

    labels = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {"Code", "activity"}

    if not required.issubset(labels.columns):
        raise ValueError(
            f"{path.name}: required columns are "
            f"{required}."
        )

    labels = labels[
        ["Code", "activity"]
    ].copy()

    labels["Code"] = normalize_code(
        labels["Code"]
    )

    labels["activity"] = pd.to_numeric(
        labels["activity"],
        errors="coerce",
    )

    validate_unique_codes(
        labels,
        path.name,
    )

    if labels["activity"].isna().any():
        raise ValueError(
            f"{path.name}: activity contains "
            f"missing or non-numeric values."
        )

    unexpected = (
        set(labels["activity"].unique())
        - {0, 1}
    )

    if unexpected:
        raise ValueError(
            f"{path.name}: activity must contain only "
            f"0 and 1. Found: {unexpected}"
        )

    labels["activity"] = (
        labels["activity"]
        .astype(np.int8)
    )

    return labels


def load_blind_codes(
    path: Path,
) -> pd.DataFrame:
    """
    Load and validate blind-set codes.
    """
    check_file_exists(path)

    blind = pd.read_csv(
        path,
        low_memory=False,
    )

    if "Code" not in blind.columns:
        raise ValueError(
            f"{path.name}: column 'Code' is missing."
        )

    blind = blind[["Code"]].copy()

    blind["Code"] = normalize_code(
        blind["Code"]
    )

    validate_unique_codes(
        blind,
        path.name,
    )

    return blind


def select_master_rows(
    master: pd.DataFrame,
    codes: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Select master rows while preserving the original code order.
    """
    selected = codes.merge(
        master,
        on="Code",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_codes = selected.loc[
        selected["_merge"] == "left_only",
        "Code",
    ].tolist()

    if missing_codes:
        raise ValueError(
            f"{source_name}: {len(missing_codes)} codes "
            f"were not found in component tables. "
            f"Examples: {missing_codes[:20]}"
        )

    selected = selected.drop(
        columns="_merge"
    )

    return selected


# ============================================================
# 10. MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("BUILDING RAW STOICHIOMETRY-AWARE MASTER DATASETS")
    print("=" * 72)

    # Load component tables
    A = load_component(
        A_PATH,
        prefix="A",
        val_col="val1",
    )

    B = load_component(
        B_PATH,
        prefix="B",
        val_col="val2",
    )

    C = load_component(
        C_PATH,
        prefix="C",
        val_col="val3",
    )

    D = load_component(
        D_PATH,
        prefix="D",
        val_col="val4",
    )

    # Merge components
    master = merge_components(
        [A, B, C, D]
    )

    print(
        f"\nMerged table shape: {master.shape}"
    )

    # Add stoichiometric representation
    master = add_stoichiometry_features(
        master
    )

    descriptor_bases = get_descriptor_bases(
        master
    )

    shared_ab_bases = get_shared_ab_bases(
        master
    )

    print(
        f"\nAll descriptor bases: "
        f"{len(descriptor_bases)}"
    )

    print(
        f"Shared A/B descriptor bases: "
        f"{len(shared_ab_bases)}"
    )

    if descriptor_bases:
        print(
            "First 20 descriptor bases:",
            descriptor_bases[:20],
        )

    # Add mixture-level representations
    master = add_arithmetic_mean_descriptors(
        master,
        descriptor_bases,
    )

    master = add_weighted_mixture_descriptors(
        master,
        descriptor_bases,
    )

    master = add_ab_interaction_descriptors(
        master,
        shared_ab_bases,
        ratio_mode=PAIRWISE_RATIO_MODE,
        eps=EPS,
    )

    print(
        f"\nFinal master shape: {master.shape}"
    )

    print("\nMixtures by number of components:")
    print(
        master["n_components"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # Load train and blind identifiers
    labels = load_train_labels(
        TRAIN_LABELS
    )

    blind_codes = load_blind_codes(
        BLIND_CODES
    )

    # Build train dataset
    train_df = select_master_rows(
        master,
        labels[["Code"]],
        source_name="original.csv",
    )

    train_df = train_df.merge(
        labels,
        on="Code",
        how="left",
        validate="one_to_one",
    )

    # Build blind dataset
    blind_df = select_master_rows(
        master,
        blind_codes,
        source_name="blind.csv",
    )

    # Check overlap
    overlap = sorted(
        set(train_df["Code"])
        & set(blind_df["Code"])
    )

    if overlap:
        raise ValueError(
            f"Train and blind datasets overlap for "
            f"{len(overlap)} codes. "
            f"Examples: {overlap[:20]}"
        )

    # Save raw master matrices
    train_df.to_csv(
        TRAIN_OUT,
        index=False,
    )

    blind_df.to_csv(
        BLIND_OUT,
        index=False,
    )

    print("\n" + "=" * 72)
    print("SAVED")
    print("=" * 72)

    print("Train:", TRAIN_OUT)
    print("Blind:", BLIND_OUT)

    print(
        f"\nTrain shape: {train_df.shape}"
    )

    print(
        f"Blind shape: {blind_df.shape}"
    )

    print("\nActivity balance:")
    print(
        train_df["activity"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nActivity percentages:")
    print(
        (
            train_df["activity"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )

    print(
        "\nFeature cleaning, imputation, scaling, PCA, "
        "SMOTE/class weighting and XGBoost modeling "
        "must be performed later."
    )

    print("\n✅ Raw master datasets created successfully.")


if __name__ == "__main__":
    main()


