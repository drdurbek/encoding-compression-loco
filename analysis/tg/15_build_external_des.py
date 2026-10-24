
"""
BUILD Tg BLIND SET 2 FEATURE MATRIX
===================================

This script creates a stoichiometry-aware feature matrix for a new, anhydrous
Tg blind set.

Important
---------
* This script does NOT fit, tune, select, calibrate, or apply a model.
* It assumes a.csv, b.csv, c.csv, and d.csv ALREADY contain molecular
  descriptors calculated with the SAME Mordred/version/protocol as the Tg
  training dataset.  It does not calculate Mordred descriptors from .mol files.
* The output is aligned exactly to the feature columns of
  TG/master_train_features.csv.
* No water-fraction correction is used: blind2 contains no aqueous solutions.

Required files
--------------
TG/
    master_train_features.csv              # reference feature schema

TG/blind2/
    a.csv, b.csv, c.csv, d.csv             # Code, val1/2/3/4, descriptors
    blind.csv                              # Code only; no activity column required
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


from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# 0. PATHS AND FROZEN SETTINGS
# =============================================================================

TG_DIR = data_path('TG')

BLIND2_DIR = data_path('TG/blind2')

A_PATH = data_path('TG/blind2/a.csv')
B_PATH = data_path('TG/blind2/b.csv')
C_PATH = data_path('TG/blind2/c.csv')
D_PATH = data_path('TG/blind2/d.csv')
BLIND_CODES_PATH = data_path('TG/blind2/blind2.csv')

REFERENCE_TRAIN_PATH = data_path('TG/master_train_features.csv')

FEATURE_OUTPUT = data_path('TG/blind2/master_blind2_features.csv')

COMPONENTS = ("A", "B", "C", "D")
EPS = 1e-6
PAIRWISE_RATIO_MODE = "directed"  # must match the Tg training encoder

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"
TG_GROUP_COLUMN = "Core_NADES_ID"

# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================

def normalize_code(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.str.replace(r"\.0$", "", regex=True)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


def validate_unique_codes(df: pd.DataFrame, source_name: str) -> None:
    if CODE_COLUMN not in df.columns:
        raise ValueError(f"{source_name}: missing required Code column.")
    if df[CODE_COLUMN].isna().any():
        raise ValueError(f"{source_name}: contains missing Code values.")
    if df[CODE_COLUMN].duplicated().any():
        examples = df.loc[df[CODE_COLUMN].duplicated(), CODE_COLUMN].head(10).tolist()
        raise ValueError(f"{source_name}: Code values must be unique. Examples: {examples}")


def descriptor_feature_columns(reference: pd.DataFrame) -> list[str]:
    """Use the exact final training schema; metadata never enters prediction."""
    excluded = {CODE_COLUMN, TARGET_COLUMN, TG_GROUP_COLUMN}
    return [column for column in reference.columns if column not in excluded]


# =============================================================================
# 2. LOAD NEW COMPONENT-DESCRIPTOR TABLES
# =============================================================================

def load_component(path: Path, prefix: str, val_column: str) -> pd.DataFrame:
    require_file(path, f"{prefix}-component table")
    table = pd.read_csv(path, low_memory=False)
    if not {CODE_COLUMN, val_column}.issubset(table.columns):
        raise ValueError(
            f"{path.name} must contain {CODE_COLUMN!r} and {val_column!r}. "
            f"Found: {table.columns.tolist()[:20]}"
        )
    table = table.copy()
    table[CODE_COLUMN] = normalize_code(table[CODE_COLUMN])
    validate_unique_codes(table, path.name)

    rename = {
        column: f"{prefix}_{column}"
        for column in table.columns
        if column not in {CODE_COLUMN, val_column}
    }
    table = table.rename(columns=rename).rename(columns={val_column: f"{prefix}_val"})
    for column in table.columns:
        if column != CODE_COLUMN:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    if (table[f"{prefix}_val"] < 0).any():
        raise ValueError(f"{path.name}: negative stoichiometric coefficient found.")
    return table


def merge_components(tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = tables[0].copy()
    for table in tables[1:]:
        merged = merged.merge(table, on=CODE_COLUMN, how="outer", validate="one_to_one")
    return merged.sort_values(CODE_COLUMN).reset_index(drop=True)


# =============================================================================
# 3. STOICHIOMETRY-AWARE MIXTURE ENCODING
# =============================================================================

def add_stoichiometry_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for component in COMPONENTS:
        val_column = f"{component}_val"
        if val_column not in out:
            out[val_column] = np.nan
        out[val_column] = pd.to_numeric(out[val_column], errors="coerce")
        out[f"{component}_val_filled"] = out[val_column].fillna(0.0).astype(float)
        out[f"has_{component}"] = (out[f"{component}_val_filled"] > 0).astype(np.int8)
    out["n_components"] = out[[f"has_{c}" for c in COMPONENTS]].sum(axis=1).astype(np.int8)
    out["val_total"] = out[[f"{c}_val_filled" for c in COMPONENTS]].sum(axis=1)
    if (out["val_total"] <= 0).any():
        codes = out.loc[out["val_total"] <= 0, CODE_COLUMN].head(10).tolist()
        raise ValueError(f"At least one blind mixture has non-positive total stoichiometry: {codes}")
    for component in COMPONENTS:
        out[f"{component}_val_frac"] = out[f"{component}_val_filled"] / out["val_total"]
    return out


def descriptor_bases(df: pd.DataFrame) -> list[str]:
    excluded = {"val", "val_filled", "val_frac"}
    bases: set[str] = set()
    for component in COMPONENTS:
        prefix = f"{component}_"
        for column in df.columns:
            if column.startswith(prefix):
                base = column[len(prefix):]
                if base not in excluded and not base.startswith("val"):
                    bases.add(base)
    return sorted(bases)


def shared_ab_bases(df: pd.DataFrame) -> list[str]:
    return sorted([
        column[2:] for column in df.columns
        if column.startswith("A_")
        and column[2:] not in {"val", "val_filled", "val_frac"}
        and not column[2:].startswith("val")
        and f"B_{column[2:]}" in df.columns
    ])


def add_mixture_descriptors(df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    """Generate avg_ and mix_ descriptors exactly from component values."""
    out = df.copy()
    average, weighted = {}, {}
    denominator = out["val_total"].replace(0.0, np.nan)

    for base in bases:
        sum_unweighted = pd.Series(0.0, index=out.index)
        sum_weighted = pd.Series(0.0, index=out.index)
        valid = pd.Series(True, index=out.index)
        for component in COMPONENTS:
            coefficient = out[f"{component}_val_filled"]
            present = coefficient > 0
            column = f"{component}_{base}"
            if column not in out.columns:
                valid &= ~present
                continue
            values = pd.to_numeric(out[column], errors="coerce")
            valid &= ~(present & values.isna())
            sum_unweighted += values.fillna(0.0).where(present, 0.0)
            sum_weighted += coefficient * values.fillna(0.0)

        avg_value = sum_unweighted / out["n_components"].replace(0, np.nan)
        mix_value = sum_weighted / denominator
        avg_value.loc[~valid] = np.nan
        mix_value.loc[~valid] = np.nan
        average[f"avg_{base}"] = avg_value
        weighted[f"mix_{base}"] = mix_value

    return pd.concat([out, pd.DataFrame(average, index=out.index), pd.DataFrame(weighted, index=out.index)], axis=1)


def ab_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    if PAIRWISE_RATIO_MODE == "directed":
        result = a / (b + EPS)
    elif PAIRWISE_RATIO_MODE == "symmetric":
        result = pd.concat([a.abs(), b.abs()], axis=1).max(axis=1) / (
            pd.concat([a.abs(), b.abs()], axis=1).min(axis=1) + EPS
        )
    else:
        raise ValueError("PAIRWISE_RATIO_MODE must be 'directed' or 'symmetric'.")
    return result.mask(a.isna() | b.isna())


def add_ab_descriptors(df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    out = df.copy()
    generated = {}
    for base in bases:
        a, b = pd.to_numeric(out[f"A_{base}"], errors="coerce"), pd.to_numeric(out[f"B_{base}"], errors="coerce")
        generated[f"AB_avg_{base}"] = (a + b) / 2.0
        generated[f"AB_diff_{base}"] = (a - b).abs()
        generated[f"AB_prod_{base}"] = a * b
        generated[f"AB_ratio_{base}"] = ab_ratio(a, b)
    a, b = pd.to_numeric(out["A_val"], errors="coerce"), pd.to_numeric(out["B_val"], errors="coerce")
    generated.update({
        "AB_avg_val": (a + b) / 2.0,
        "AB_diff_val": (a - b).abs(),
        "AB_prod_val": a * b,
        "AB_ratio_val": ab_ratio(a, b),
    })
    return pd.concat([out, pd.DataFrame(generated, index=out.index)], axis=1)


# =============================================================================
# 4. BLIND CODES AND ALIGNMENT
# =============================================================================

def load_blind_codes() -> pd.DataFrame:
    require_file(BLIND_CODES_PATH, "blind2 code table")
    blind = pd.read_csv(BLIND_CODES_PATH, low_memory=False)
    if CODE_COLUMN not in blind.columns:
        raise ValueError("blind.csv must contain Code.")
    blind = blind[[CODE_COLUMN]].copy()
    blind[CODE_COLUMN] = normalize_code(blind[CODE_COLUMN])
    validate_unique_codes(blind, BLIND_CODES_PATH.name)
    return blind


def select_blind_rows(master: pd.DataFrame, blind_codes: pd.DataFrame) -> pd.DataFrame:
    selected = blind_codes.merge(master, on=CODE_COLUMN, how="left", validate="one_to_one", indicator=True)
    missing = selected.loc[selected["_merge"] == "left_only", CODE_COLUMN].tolist()
    if missing:
        raise ValueError(f"blind2 Codes not found in the component tables: {missing}")
    return selected.drop(columns="_merge")


def align_to_training_schema(raw: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    reference_features = descriptor_feature_columns(reference)
    missing = [column for column in reference_features if column not in raw.columns]
    if missing:
        raise ValueError(
            "blind2 encoding does not match the Tg training schema. Missing "
            f"{len(missing)} columns, e.g. {missing[:20]}.\n"
            "Check that all component descriptor tables were generated with the same Mordred protocol."
        )
    output = raw[[CODE_COLUMN]].copy()
    X = raw.loc[:, reference_features].apply(pd.to_numeric, errors="coerce")
    output = pd.concat([output, X], axis=1)
    return output


# =============================================================================
# 5. MAIN
# =============================================================================

def main() -> None:
    for path, name in [
        (A_PATH, "a.csv"), (B_PATH, "b.csv"), (C_PATH, "c.csv"), (D_PATH, "d.csv"),
        (REFERENCE_TRAIN_PATH, "Tg training master"),
    ]:
        require_file(path, name)

    print("=" * 100)
    print("TG BLIND2 — STOICHIOMETRY-AWARE FEATURE ENCODING")
    print("=" * 100)

    reference = pd.read_csv(REFERENCE_TRAIN_PATH, low_memory=False)
    if {CODE_COLUMN, TARGET_COLUMN} - set(reference.columns):
        raise ValueError("Reference Tg training master must contain Code and activity.")

    components = [
        load_component(A_PATH, "A", "val1"),
        load_component(B_PATH, "B", "val2"),
        load_component(C_PATH, "C", "val3"),
        load_component(D_PATH, "D", "val4"),
    ]
    master = add_stoichiometry_columns(merge_components(components))
    bases = descriptor_bases(master)
    ab_bases = shared_ab_bases(master)
    if not bases:
        raise ValueError(
            "No molecular descriptor columns were found in a.csv–d.csv. "
            "First run the same Mordred-descriptor-generation step used for Tg training."
        )
    print(f"New blind2 mixtures: {len(master)} | descriptor bases: {len(bases)} | shared A/B bases: {len(ab_bases)}")
    master = add_mixture_descriptors(master, bases)
    master = add_ab_descriptors(master, ab_bases)

    blind_codes = load_blind_codes()
    raw = select_blind_rows(master, blind_codes)
    features = align_to_training_schema(raw, reference)
    features.to_csv(FEATURE_OUTPUT, index=False)

    print("\nSaved:")
    print(" -", FEATURE_OUTPUT)
    print(f"\nFeature matrix shape: {features.shape}")
    print("No water-fraction correction, Tg values, or model predictions were used.")


if __name__ == "__main__":
    main()

