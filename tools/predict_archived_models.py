"""Predict with archived Final60 models without retraining or changing selection.

The classification model uses the portable XGBoost JSON export; its imputer is
reconstructed from its recorded training rows. Tg uses the author RF bundle.
Only use Tg joblib bundles from a trusted source. Input must be an already
encoded endpoint-specific matrix (concentration-scaled for Tg).
"""
from project_paths import ProjectLayout
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--endpoint', required=True, choices=['classification', 'tg'])
    p.add_argument('--purpose', choices=['heldout', 'deployment'], default='deployment')
    p.add_argument('--input', type=Path, help='Encoded input; default is the saved held-out/blind matrix.')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    root = a.data_root.resolve()
    paths = ProjectLayout(root)
    out = a.out.resolve()
    if root == out or root in out.parents:
        p.error('--out must be outside data-root')
    prefix = 'TG/' if a.endpoint == 'tg' else ''
    if a.endpoint == 'classification':
        from xgboost import XGBClassifier
        features = pd.read_csv(paths.path('Final60_Internal_Evaluation/Final60_selected_descriptors.csv')).feature.tolist()
        train = pd.read_csv(paths.path('master_train_features.csv'), usecols=['Code'] + features, low_memory=False)
        splits = pd.read_csv(paths.path('Final_clean_split_seed42.csv')).set_index('Code').split
        fit = np.ones(len(train), dtype=bool) if a.purpose == 'deployment' else train.Code.map(splits).eq('Train').to_numpy()
        raw = train[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='median').fit(raw.loc[fit])
        model = XGBClassifier()
        name = 'Final60_full_dataset_xgboost_model.json' if a.purpose == 'deployment' else 'Final60_train_subset_xgboost_model.json'
        model.load_model(paths.path('Final60_Internal_Evaluation') / name)
    else:
        import joblib
        name = 'TG_Final60_full_dataset_RF_bundle.joblib' if a.purpose == 'deployment' else 'TG_Final60_outer_train_RF_bundle.joblib'
        bundle = joblib.load(paths.path('TG/TG_Final60_RF_Internal_Evaluation') / name)
        model, imputer, features = (bundle['model'], bundle['imputer'], bundle['selected_features'])
    source = a.input or paths.path(prefix + ('master_blind_features.csv' if a.purpose == 'deployment' else 'master_train_features.csv'))
    data = pd.read_csv(source, usecols=['Code'] + features, low_memory=False)
    if a.input is None and a.purpose == 'heldout':
        split_path = 'Final_clean_split_seed42.csv' if a.endpoint == 'classification' else 'TG/TG_clean_split_seed42.csv'
        split = pd.read_csv(paths.path(split_path)).set_index('Code').split
        fit = data.Code.map(split).str.lower().isin(['train', 'outer_train', 'training'])
        data = data.loc[~fit].copy()
    if not data.Code.is_unique or data.Code.isna().any():
        raise ValueError('Code must be complete and unique')
    raw = data[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    X = imputer.transform(raw)
    result = pd.DataFrame({'Code': data.Code})
    if a.endpoint == 'classification':
        probability = model.predict_proba(X.astype(np.float32))[:, 1]
        result['probability_NADES'] = probability
        result['threshold_used'] = 0.42
        result['predicted_activity'] = (probability >= 0.42).astype(int)
    else:
        result['predicted_Tg_K'] = model.predict(X)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f'Saved {len(result)} predictions to {out}')
if __name__ == '__main__':
    main()
