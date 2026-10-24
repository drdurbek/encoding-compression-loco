
"""Create concentration-aware Tg master matrices.

For every system:
    S_solution = (1 - water_frc) * S_nominal

The script preserves Code and, in the training matrix, activity unchanged.
It does not append water_frc or NADES_fraction as new model features.
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
# 1. PATHS AND SETTINGS
# ============================================================

DATA = data_path('TG')

TRAIN_MASTER_IN = data_path('TG/master_train_features1.csv')
BLIND_MASTER_IN = data_path('TG/master_blind_features1.csv')

TRAIN_WATER_IN = data_path('TG/original_water_frc.csv')
BLIND_WATER_IN = data_path('TG/blind_water_frc.csv')

TRAIN_MASTER_OUT = data_path('TG/master_train_features.csv')
BLIND_MASTER_OUT = data_path('TG/master_blind_features.csv')

CODE_COLUMN = "Code"
WATER_FRACTION_COLUMN = "water_frc"
TARGET_COLUMN = "activity"  # Legacy name; contains continuous Tg values.


# ============================================================
# 2. VALIDATION HELPERS
# ============================================================

def normalize_code(series: pd.Series) -> pd.Series:
    """Normalise codes while preserving missing values."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def load_master(path: Path, has_target: bool) -> pd.DataFrame:
    """Load one unscaled master matrix and validate essential columns."""
    if not path.exists():
        raise FileNotFoundError(f"Master matrix was not found:\n{path}")

    df = pd.read_csv(path, low_memory=False)

    required = {CODE_COLUMN}
    if has_target:
        required.add(TARGET_COLUMN)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {sorted(missing)}"
        )

    df[CODE_COLUMN] = normalize_code(df[CODE_COLUMN])

    if df[CODE_COLUMN].isna().any():
        raise ValueError(f"{path.name}: Code contains missing values.")

    if df[CODE_COLUMN].duplicated().any():
        examples = (
            df.loc[df[CODE_COLUMN].duplicated(keep=False), CODE_COLUMN]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{path.name}: Code must be unique. Examples: {examples}"
        )

    return df


def load_water_fractions(path: Path) -> pd.DataFrame:
    """Load Code/water_frc table and verify that fractions are in [0, 1)."""
    if not path.exists():
        raise FileNotFoundError(f"Water-fraction file was not found:\n{path}")

    df = pd.read_csv(path, low_memory=False)
    required = {CODE_COLUMN, WATER_FRACTION_COLUMN}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: required columns are {sorted(required)}; "
            f"missing {sorted(missing)}"
        )

    df = df[[CODE_COLUMN, WATER_FRACTION_COLUMN]].copy()
    df[CODE_COLUMN] = normalize_code(df[CODE_COLUMN])
    df[WATER_FRACTION_COLUMN] = pd.to_numeric(
        df[WATER_FRACTION_COLUMN],
        errors="coerce",
    )

    if df[CODE_COLUMN].isna().any() or df[CODE_COLUMN].duplicated().any():
        raise ValueError(f"{path.name}: Code must be present and unique.")

    if df[WATER_FRACTION_COLUMN].isna().any():
        raise ValueError(
            f"{path.name}: water_frc contains missing or non-numeric values."
        )

    invalid = (
        (df[WATER_FRACTION_COLUMN] < 0.0)
        | (df[WATER_FRACTION_COLUMN] >= 1.0)
    )

    if invalid.any():
        examples = df.loc[
            invalid,
            [CODE_COLUMN, WATER_FRACTION_COLUMN],
        ].head(10).to_dict("records")
        raise ValueError(
            "water_frc must be a fraction in [0, 1), not a percentage. "
            f"Invalid examples: {examples}"
        )

    return df


# ============================================================
# 3. CONCENTRATION SCALING
# ============================================================

def scale_master_by_nades_fraction(
    master: pd.DataFrame,
    water_fractions: pd.DataFrame,
    has_target: bool,
    source_name: str,
) -> pd.DataFrame:
    """Multiply every descriptor by c_NADES = 1 - water_frc."""
    merged = master.merge(
        water_fractions,
        on=CODE_COLUMN,
        how="left",
        validate="one_to_one",
    )

    missing_water = merged[WATER_FRACTION_COLUMN].isna()
    if missing_water.any():
        examples = merged.loc[
            missing_water,
            CODE_COLUMN,
        ].head(20).tolist()
        raise ValueError(
            f"{source_name}: water_frc is missing for "
            f"{int(missing_water.sum())} master rows. Examples: {examples}"
        )

    nades_fraction = 1.0 - merged[WATER_FRACTION_COLUMN]

    non_descriptor_columns = {CODE_COLUMN, WATER_FRACTION_COLUMN}
    if has_target:
        non_descriptor_columns.add(TARGET_COLUMN)

    descriptor_columns = [
        column
        for column in master.columns
        if column not in non_descriptor_columns
    ]

    non_numeric = [
        column
        for column in descriptor_columns
        if not pd.api.types.is_numeric_dtype(merged[column])
    ]

    if non_numeric:
        raise ValueError(
            f"{source_name}: non-numeric columns were found among "
            f"descriptors: {non_numeric[:20]}"
        )

    # This is the intended solution-level encoding:
    # S_solution = c_NADES × S_nominal.
    #
    # Cast first: several structural/stoichiometric descriptors are
    # integer columns in the raw master matrix but become fractional
    # after concentration scaling.  Casting prevents pandas from
    # emitting an incompatible-dtype FutureWarning.
    scaled_descriptors = (
        merged[descriptor_columns]
        .astype(np.float64)
        .multiply(
            nades_fraction.astype(np.float64),
            axis=0,
        )
    )

    merged[descriptor_columns] = scaled_descriptors

    # Keep the original column order exactly; water_frc is audit-only.
    out = merged.loc[:, master.columns].copy()

    if list(out.columns) != list(master.columns):
        raise RuntimeError("Output column order changed unexpectedly.")

    print(
        f"{source_name}: {len(out)} rows | "
        f"{len(descriptor_columns)} descriptors scaled | "
        f"c_NADES range = [{nades_fraction.min():.4f}, "
        f"{nades_fraction.max():.4f}]"
    )

    return out


# ============================================================
# 4. RUN
# ============================================================

def main() -> None:
    train_master = load_master(
        TRAIN_MASTER_IN,
        has_target=True,
    )

    blind_master = load_master(
        BLIND_MASTER_IN,
        has_target=False,
    )

    train_water = load_water_fractions(TRAIN_WATER_IN)
    blind_water = load_water_fractions(BLIND_WATER_IN)

    train_scaled = scale_master_by_nades_fraction(
        master=train_master,
        water_fractions=train_water,
        has_target=True,
        source_name="Training master",
    )

    blind_scaled = scale_master_by_nades_fraction(
        master=blind_master,
        water_fractions=blind_water,
        has_target=False,
        source_name="Blind master",
    )

    expected_blind_columns = [
        column
        for column in train_scaled.columns
        if column != TARGET_COLUMN
    ]

    if list(blind_scaled.columns) != expected_blind_columns:
        raise ValueError(
            "Train and blind matrices do not have the same predictor columns."
        )

    train_scaled.to_csv(TRAIN_MASTER_OUT, index=False)
    blind_scaled.to_csv(BLIND_MASTER_OUT, index=False)

    print("\nSaved concentration-aware Tg matrices:")
    print(" -", TRAIN_MASTER_OUT)
    print(" -", BLIND_MASTER_OUT)
    print("\nCode and activity were preserved without scaling.")


if __name__ == "__main__":
    main()

