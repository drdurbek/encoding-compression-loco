
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
# FINAL60: 5-FOLD POST-SELECTION OOF PROBABILITIES
# For calibration diagnostics only
#
# Requires:
#   X            -- full cleaned descriptor DataFrame
#   y            -- full binary target, 0/non-NADES and 1/NADES
#   selected_features
# ============================================================

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# ------------------------------------------------------------
# 1. FROZEN FINAL60 MODEL SETTINGS
# ------------------------------------------------------------

SEED = 42
MODEL_SEED = 82

N_ESTIMATORS = 300
MAX_DEPTH = 5
LEARNING_RATE = 0.05

# ------------------------------------------------------------
# 2. RECONSTRUCT FULL DATA FROM THE EXISTING HOLD-OUT SPLIT
# ------------------------------------------------------------

if "X_train_raw" not in globals():
    raise NameError(
        "X_train_raw is missing. Run the main Final60 held-out "
        "training script first."
    )

if "X_test_raw" not in globals():
    raise NameError(
        "X_test_raw is missing. Run the main Final60 held-out "
        "training script first."
    )

if "y_train" not in globals():
    raise NameError(
        "y_train is missing. Run the main Final60 held-out "
        "training script first."
    )

if "y_test" not in globals():
    raise NameError(
        "y_test is missing. Run the main Final60 held-out "
        "training script first."
    )

X_oof_raw = (
    pd.concat(
        [
            X_train_raw,
            X_test_raw,
        ],
        axis=0,
    )
    .loc[:, selected_features]
    .copy()
)

y_oof = np.concatenate(
    [
        np.asarray(y_train, dtype=int),
        np.asarray(y_test, dtype=int),
    ]
)

if len(X_oof_raw) != len(y_oof):
    raise ValueError(
        "Reconstructed X_oof_raw and y_oof have different lengths."
    )

print(
    f"OOF data reconstructed: {len(y_oof)} samples | "
    f"non-NADES = {(y_oof == 0).sum()} | "
    f"NADES = {(y_oof == 1).sum()}"
)
# ------------------------------------------------------------
# 3. FIVE-FOLD OOF PREDICTIONS
# ------------------------------------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

proba_oof = np.full(
    len(y_oof),
    np.nan,
    dtype=float,
)

for fold, (train_index, validation_index) in enumerate(
    skf.split(X_oof_raw, y_oof),
    start=1,
):
    X_fold_train = X_oof_raw.iloc[
        train_index
    ]

    X_fold_validation = X_oof_raw.iloc[
        validation_index
    ]

    y_fold_train = y_oof[
        train_index
    ]

    # Imputation fitted only on the four training folds.
    fold_imputer = SimpleImputer(
        strategy="median"
    )

    X_fold_train_imp = fold_imputer.fit_transform(
        X_fold_train
    )

    X_fold_validation_imp = fold_imputer.transform(
        X_fold_validation
    )

    n_nades_train = int(
        (y_fold_train == 1).sum()
    )

    n_non_nades_train = int(
        (y_fold_train == 0).sum()
    )

    scale_pos_weight = (
        n_non_nades_train
        / max(n_nades_train, 1)
    )

    fold_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=0.80,
        colsample_bytree=0.80,
        random_state=MODEL_SEED,
        eval_metric="logloss",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
    )

    fold_model.fit(
        X_fold_train_imp,
        y_fold_train,
    )

    proba_oof[
        validation_index
    ] = fold_model.predict_proba(
        X_fold_validation_imp
    )[:, 1]

    print(
        f"Fold {fold}/5 completed | "
        f"validation n = {len(validation_index)}"
    )

if not np.isfinite(proba_oof).all():
    raise RuntimeError(
        "Some OOF probabilities were not generated."
    )

print("\nFinal60 post-selection OOF results:")

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_oof, proba_oof):.4f}"
)

print(
    f"AP for NADES: "
    f"{average_precision_score(y_oof, proba_oof):.4f}"
)


