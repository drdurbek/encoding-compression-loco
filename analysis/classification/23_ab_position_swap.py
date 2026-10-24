
"""
AB POSITION-SWAP CONTROL FOR THE NADES CLASSIFICATION ENCODING
==============================================================

Purpose
-------
This is an explanatory control, not a new performance-optimisation run.
It tests whether the apparent dominance of Component_B descriptors is tied
to the *B slot* in the database rather than to an intrinsic property of the
chemical component labelled "B".

For every mixture, the script exchanges A and B consistently:

    A descriptors <-> B descriptors
    A_val         <-> B_val
    A/B difference -> B/A difference = -(A/B difference)
    A/B ratio      -> B/A ratio

Symmetric mixture operators (mix_, avg_, AB_avg_, AB_prod_, AB_mult_) are
unchanged by construction. C and D information is untouched.

IMPORTANT
---------
Run the *same* train-only prefilter and DAGA subset-size sweep afterwards,
with exactly the same fixed split and seeds.  Only DATA_PATH changes.
Do not use the held-out test or blind set to choose between the original and
swapped analyses.  Compare the selected-descriptor family/position counts
and OOF metrics only.

Expected interpretation
-----------------------
If the original Component_B enrichment transfers to Component_A after this
swap while OOF performance remains similar, the result is positional:
the variable component slot, not the letter "B", carries the signal.
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
import re

import numpy as np
import pandas as pd


# ============================================================================
# 1. PATHS
# ============================================================================

BASE = data_path('')

# This must be the exact encoded labelled dataset that fed the original,
# leakage-free classification prefilter.
INPUT_PATH = data_path('master_train_features.csv')

OUTPUT_PATH = data_path('master_train_features_AB_position_swapped.csv')
MANIFEST_PATH = data_path('AB_position_swap_manifest.csv')
QC_PATH = data_path('AB_position_swap_quality_control.csv')


# ============================================================================
# 2. SETTINGS
# ============================================================================

CODE_COLUMN = "Code"
TARGET_COLUMN = "activity"

# Direct component-descriptor pairs are recognised as A_<suffix> and
# B_<suffix>.  A_ and B_ prefixes are deliberately anchored so that AB_
# interaction columns are never mistaken for component columns.
COMPONENT_A_PREFIX = "A_"
COMPONENT_B_PREFIX = "B_"

# Optional metadata columns.  If present, they are swapped too.  Absence is
# harmless and is recorded in the manifest only through the component pairing.
OPTIONAL_METADATA_PAIRS = [
    ("A", "B"),
    ("A_name", "B_name"),
    ("A_component", "B_component"),
    ("A_SMILES", "B_SMILES"),
    ("A_InChIKey", "B_InChIKey"),
    ("A_CAS", "B_CAS"),
]

# Directed A/B interaction operators.  Extend these lists only if the
# original encoder created additional directional operators.
DIFFERENCE_PREFIXES = (
    "AB_diff_",
    "AB_difference_",
    "AB_sub_",
)

RATIO_PREFIXES = (
    "AB_ratio_",
)

# The original encoder must have used the ordinary direct ratio A/B for an
# AB_ratio_ column.  In that case the swapped value is B/A = 1/(A/B).
# The script refuses to invert a zero/non-finite ratio because that would not
# reproduce the original encoder faithfully; rebuild that ratio from raw
# component descriptors in the encoder instead.
REQUIRE_FINITE_NONZERO_RATIOS = True


# ============================================================================
# 3. HELPERS
# ============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found:\n{path}")


def is_protected(column: str) -> bool:
    return column in {CODE_COLUMN, TARGET_COLUMN}


def component_pairs(columns: pd.Index) -> list[tuple[str, str]]:
    """Return all matched A_<suffix> <-> B_<suffix> pairs."""
    columns_set = set(columns)
    pairs: list[tuple[str, str]] = []

    for column in columns:
        if not column.startswith(COMPONENT_A_PREFIX):
            continue

        suffix = column[len(COMPONENT_A_PREFIX):]
        paired_column = f"{COMPONENT_B_PREFIX}{suffix}"

        if paired_column in columns_set:
            pairs.append((column, paired_column))

    return pairs


def directed_columns(columns: pd.Index, prefixes: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in columns
        if any(column.startswith(prefix) for prefix in prefixes)
    ]


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Convert a descriptor column safely and fail on non-numeric values."""
    series = pd.to_numeric(frame[column], errors="coerce")

    original_not_missing = frame[column].notna()
    became_missing = original_not_missing & series.isna()

    if became_missing.any():
        examples = frame.loc[became_missing, column].head(5).tolist()
        raise ValueError(
            f"{column} contains non-numeric values; examples: {examples}"
        )

    return series.astype(float)


def append_qc(
    qc_rows: list[dict[str, object]],
    rule: str,
    column: str,
    partner: str | None,
    maximum_absolute_error: float,
    passed: bool,
) -> None:
    qc_rows.append(
        {
            "rule": rule,
            "column": column,
            "partner_column": partner,
            "maximum_absolute_error": maximum_absolute_error,
            "passed": passed,
        }
    )


# ============================================================================
# 4. LOAD AND VALIDATE INPUT
# ============================================================================

require_file(INPUT_PATH)

data = pd.read_csv(INPUT_PATH, low_memory=False)

required_columns = {CODE_COLUMN, TARGET_COLUMN}
missing_required = required_columns - set(data.columns)

if missing_required:
    raise ValueError(
        f"Input table lacks required columns: {sorted(missing_required)}"
    )

if data[CODE_COLUMN].duplicated().any():
    examples = data.loc[data[CODE_COLUMN].duplicated(), CODE_COLUMN].head(10).tolist()
    raise ValueError(
        f"{CODE_COLUMN} must be unique. Duplicate examples: {examples}"
    )

original = data.copy(deep=True)
swapped = data.copy(deep=True)

manifest_rows: list[dict[str, object]] = []
qc_rows: list[dict[str, object]] = []


# ============================================================================
# 5. SWAP DIRECT A/B COMPONENT INFORMATION
# ============================================================================

all_pairs = component_pairs(data.columns)

if not all_pairs:
    raise ValueError(
        "No matched A_<suffix> / B_<suffix> descriptor pairs were found. "
        "Check COMPONENT_A_PREFIX and COMPONENT_B_PREFIX."
    )

for column_a, column_b in all_pairs:
    value_a = original[column_a].copy()
    value_b = original[column_b].copy()

    swapped[column_a] = value_b
    swapped[column_b] = value_a

    manifest_rows.extend(
        [
            {
                "output_column": column_a,
                "source_column": column_b,
                "transformation": "swap_A_B_component_column",
            },
            {
                "output_column": column_b,
                "source_column": column_a,
                "transformation": "swap_A_B_component_column",
            },
        ]
    )

    same_a = swapped[column_a].equals(original[column_b])
    same_b = swapped[column_b].equals(original[column_a])

    append_qc(
        qc_rows,
        "direct_component_swap",
        column_a,
        column_b,
        0.0,
        bool(same_a and same_b),
    )


# Optional non-descriptor A/B metadata swaps.
for column_a, column_b in OPTIONAL_METADATA_PAIRS:
    if column_a not in data.columns or column_b not in data.columns:
        continue

    # A_val/B_val are already handled by the descriptor pairing loop.
    if (column_a, column_b) in all_pairs:
        continue

    swapped[column_a] = original[column_b]
    swapped[column_b] = original[column_a]

    manifest_rows.extend(
        [
            {
                "output_column": column_a,
                "source_column": column_b,
                "transformation": "swap_A_B_metadata",
            },
            {
                "output_column": column_b,
                "source_column": column_a,
                "transformation": "swap_A_B_metadata",
            },
        ]
    )

    append_qc(
        qc_rows,
        "optional_metadata_swap",
        column_a,
        column_b,
        0.0,
        bool(
            swapped[column_a].equals(original[column_b])
            and swapped[column_b].equals(original[column_a])
        ),
    )


# ============================================================================
# 6. TRANSFORM DIRECTED AB OPERATORS
# ============================================================================

# Swapping A and B reverses an oriented difference.
for column in directed_columns(data.columns, DIFFERENCE_PREFIXES):
    old_value = numeric_series(original, column)
    swapped[column] = -old_value

    observed = numeric_series(swapped, column)
    max_error = float(np.nanmax(np.abs(observed.to_numpy() + old_value.to_numpy())))
    if not np.isfinite(max_error):
        max_error = 0.0

    manifest_rows.append(
        {
            "output_column": column,
            "source_column": column,
            "transformation": "negate_oriented_AB_difference",
        }
    )
    append_qc(
        qc_rows,
        "oriented_difference",
        column,
        None,
        max_error,
        bool(np.isclose(max_error, 0.0)),
    )


# Swapping A and B inverts an oriented A/B ratio.
for column in directed_columns(data.columns, RATIO_PREFIXES):
    old_value = numeric_series(original, column)
    finite_nonzero = np.isfinite(old_value.to_numpy()) & (old_value.to_numpy() != 0.0)

    if REQUIRE_FINITE_NONZERO_RATIOS and not finite_nonzero.all():
        n_bad = int((~finite_nonzero).sum())
        raise ValueError(
            f"{column} contains {n_bad} zero or non-finite values. "
            "Do not approximate B/A by reciprocal values in this case. "
            "Create the swapped raw mixture table and rerun the original "
            "encoder so AB ratios are rebuilt with its exact safe-division rule."
        )

    new_value = pd.Series(np.nan, index=old_value.index, dtype=float)
    new_value.loc[finite_nonzero] = 1.0 / old_value.loc[finite_nonzero]
    swapped[column] = new_value

    reciprocal_product = (
        numeric_series(swapped, column).to_numpy()[finite_nonzero]
        * old_value.to_numpy()[finite_nonzero]
    )
    max_error = float(np.max(np.abs(reciprocal_product - 1.0)))

    manifest_rows.append(
        {
            "output_column": column,
            "source_column": column,
            "transformation": "invert_oriented_AB_ratio",
        }
    )
    append_qc(
        qc_rows,
        "oriented_ratio",
        column,
        None,
        max_error,
        bool(np.isclose(max_error, 0.0, atol=1e-12)),
    )


# ============================================================================
# 7. INVARIANCE AND INTEGRITY CHECKS
# ============================================================================

# These operators are symmetric under A<->B and must not change.
symmetric_prefixes = (
    "mix_",
    "avg_",
    "AB_avg_",
    "AB_prod_",
    "AB_mult_",
)

for column in [
    c for c in data.columns
    if any(c.startswith(prefix) for prefix in symmetric_prefixes)
]:
    if not swapped[column].equals(original[column]):
        raise AssertionError(
            f"Symmetric operator unexpectedly changed: {column}"
        )

if not swapped[CODE_COLUMN].equals(original[CODE_COLUMN]):
    raise AssertionError("Code column changed during the position-swap control.")

if not swapped[TARGET_COLUMN].equals(original[TARGET_COLUMN]):
    raise AssertionError("activity labels changed during the position-swap control.")

qc = pd.DataFrame(qc_rows)

if not qc.empty and not qc["passed"].all():
    failed = qc.loc[~qc["passed"]]
    raise AssertionError(
        "One or more position-swap quality checks failed:\n"
        f"{failed.to_string(index=False)}"
    )


# ============================================================================
# 8. SAVE
# ============================================================================

swapped.to_csv(OUTPUT_PATH, index=False)
pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
qc.to_csv(QC_PATH, index=False)

print("=" * 94)
print("A/B POSITION-SWAP CONTROL DATASET CREATED")
print("=" * 94)
print(f"Input rows / columns              : {original.shape[0]} / {original.shape[1]}")
print(f"Matched A<->B descriptor pairs    : {len(all_pairs)}")
print(f"Directed AB differences negated   : {len(directed_columns(data.columns, DIFFERENCE_PREFIXES))}")
print(f"Directed AB ratios inverted       : {len(directed_columns(data.columns, RATIO_PREFIXES))}")
print(f"QC checks passed                  : {int(qc['passed'].sum()) if not qc.empty else 0}/{len(qc)}")
print()
print("Saved:")
print(" -", OUTPUT_PATH)
print(" -", MANIFEST_PATH)
print(" -", QC_PATH)
print()
print("NEXT STEP — rerun the original TRAIN-ONLY prefilter and DAGA sweep with:")
print(f"DATA_PATH = {OUTPUT_PATH}")
print("Keep Final_clean_split_seed42.csv, all seeds, all hyperparameters, and")
print("the OOF-only size-selection rule unchanged. Do not use held-out/blind")
print("performance to interpret this position-swap control.")

