
"""Add leakage-control parent-NADES groups to the Tg training master matrix.

Core_NADES_ID is built from the component identities in original.csv and the
corresponding A/B/C/D stoichiometric coefficients in a.csv--d.csv.  It is
permutation-invariant for grouping only: component positions in the descriptor
encoding remain unchanged.
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


# ============================================================
# 1. PATHS
# ============================================================

DATA = data_path('TG')

ORIGINAL_PATH = data_path('TG/original.csv')
A_PATH = data_path('TG/a.csv')
B_PATH = data_path('TG/b.csv')
C_PATH = data_path('TG/c.csv')
D_PATH = data_path('TG/d.csv')

MASTER_TRAIN_PATH = data_path('TG/master_train_features.csv')
GROUP_TABLE_OUT = data_path('TG/TG_core_nades_groups.csv')

CODE_COLUMN = "Code"
GROUP_COLUMN = "Core_NADES_ID"

COMPONENT_SPECS = [
    ("A_id", A_PATH, "val1"),
    ("B_id", B_PATH, "val2"),
    ("C_id", C_PATH, "val3"),
    ("D_id", D_PATH, "val4"),
]


# ============================================================
# 2. HELPERS
# ============================================================

def normalize_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def build_core_nades_id(row: pd.Series) -> str:
    """Return a composition-normalised, order-invariant parent-NADES ID."""
    pairs: list[tuple[str, float]] = []

    for name_column, _, value_column in COMPONENT_SPECS:
        name = row[name_column]
        coefficient = pd.to_numeric(row[value_column], errors="coerce")

        if pd.notna(name) and pd.notna(coefficient) and coefficient > 0:
            name = str(name).strip()
            if name and name.lower() != "nan":
                pairs.append((name, float(coefficient)))

    total = sum(coefficient for _, coefficient in pairs)
    if total <= 0:
        raise ValueError(
            f"Code {row[CODE_COLUMN]} has no valid structural components."
        )

    # Sorting makes AB and BA the same physical core formulation for splitting.
    # Ratios 1:2 and 2:4 also become the same formulation after normalisation.
    return "|".join(
        f"{name}:{coefficient / total:.8f}"
        for name, coefficient in sorted(
            pairs,
            key=lambda item: item[0].casefold(),
        )
    )


# ============================================================
# 3. BUILD GROUP TABLE
# ============================================================

def main() -> None:
    for path in [
        ORIGINAL_PATH,
        A_PATH,
        B_PATH,
        C_PATH,
        D_PATH,
        MASTER_TRAIN_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    original = pd.read_csv(ORIGINAL_PATH, low_memory=False)
    required_original = {CODE_COLUMN, "A_id", "B_id", "C_id", "D_id"}
    missing = required_original - set(original.columns)
    if missing:
        raise ValueError(
            f"original.csv is missing columns: {sorted(missing)}"
        )

    original = original.copy()
    original[CODE_COLUMN] = normalize_code(original[CODE_COLUMN])

    if original[CODE_COLUMN].isna().any() or original[CODE_COLUMN].duplicated().any():
        raise ValueError("original.csv requires unique, non-missing Code values.")

    group_data = original[
        [CODE_COLUMN, "A_id", "B_id", "C_id", "D_id"]
    ].copy()

    for _, path, value_column in COMPONENT_SPECS:
        values = pd.read_csv(path, usecols=[CODE_COLUMN, value_column])
        values[CODE_COLUMN] = normalize_code(values[CODE_COLUMN])
        values[value_column] = pd.to_numeric(values[value_column], errors="coerce")

        if values[CODE_COLUMN].duplicated().any():
            raise ValueError(f"{path.name}: Code must be unique.")

        group_data = group_data.merge(
            values,
            on=CODE_COLUMN,
            how="left",
            validate="one_to_one",
        )

    group_data[GROUP_COLUMN] = group_data.apply(
        build_core_nades_id,
        axis=1,
    )

    groups = group_data[[CODE_COLUMN, GROUP_COLUMN]].copy()
    groups.to_csv(GROUP_TABLE_OUT, index=False)

    master = pd.read_csv(MASTER_TRAIN_PATH, low_memory=False)
    master[CODE_COLUMN] = normalize_code(master[CODE_COLUMN])

    if GROUP_COLUMN in master.columns:
        master = master.drop(columns=[GROUP_COLUMN])

    master = master.merge(
        groups,
        on=CODE_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if master[GROUP_COLUMN].isna().any():
        missing_codes = master.loc[
            master[GROUP_COLUMN].isna(),
            CODE_COLUMN,
        ].head(20).tolist()
        raise ValueError(
            "Some master rows could not be assigned Core_NADES_ID. "
            f"Examples: {missing_codes}"
        )

    master.to_csv(MASTER_TRAIN_PATH, index=False)

    group_sizes = groups[GROUP_COLUMN].value_counts()

    print("=" * 80)
    print("TG CORE NADES GROUPS CREATED")
    print("=" * 80)
    print(f"Rows                 : {len(groups)}")
    print(f"Unique parent NADES  : {groups[GROUP_COLUMN].nunique()}")
    print("\nNumber of groups by concentration-series length:")
    print(group_sizes.value_counts().sort_index())
    print("\nSaved:")
    print(" -", GROUP_TABLE_OUT)
    print(" -", MASTER_TRAIN_PATH)
    print(
        "\nCore_NADES_ID is metadata for grouped validation; "
        "it is not a model descriptor."
    )


if __name__ == "__main__":
    main()

