"""Audit supplied data and saved predictions; optionally refit the frozen models.

Does not repeat descriptor selection, bootstrap CIs, nested LOCO or cross-fitting.
Writes only to --out. Never modifies source data or stored author results.
"""
from project_paths import ProjectLayout
import argparse
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, r2_score, mean_absolute_error, confusion_matrix

def regression(y, p):
    y, p = (np.asarray(y, float), np.asarray(p, float))
    return dict(n=len(y), r2=float(r2_score(y, p)), rmse_K=float(np.sqrt(np.mean((y - p) ** 2))), mae_K=float(mean_absolute_error(y, p)))

def classification(y, p, prediction):
    y, p, prediction = (np.asarray(y, int), np.asarray(p, float), np.asarray(prediction, int))
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(n=len(y), roc_auc=float(roc_auc_score(y, p)), ap_nades=float(average_precision_score(y, p)), ap_non_nades=float(average_precision_score(1 - y, 1 - p)), balanced_accuracy=float(balanced_accuracy_score(y, prediction)), mcc=float(matthews_corrcoef(y, prediction)), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--refit', action='store_true', help='Refit four fixed-feature models, not feature selection.')
    args = parser.parse_args()
    root, out = (args.data_root.resolve(), args.out.resolve())
    paths = ProjectLayout(root)
    if out == root or root in out.parents:
        parser.error('--out must be outside data-root')
    out.mkdir(parents=True, exist_ok=True)
    checks, observations, metrics = ([], {}, [])

    def check(name, condition, **detail):
        checks.append(dict(name=name, passed=bool(condition), **detail))

    def read(path, **kw):
        return pd.read_csv(paths.path(path), low_memory=False, **kw)

    def pred_metrics(path, ycol, pcol, groups=(), binary=False, predcol=None, threshold=0.42):
        d = read(path)
        grouped = d.groupby(list(groups), dropna=False) if groups else [((), d)]
        for key, g in grouped:
            key = key if isinstance(key, tuple) else (key,)
            info = dict(source=path, **{k: str(v) for k, v in zip(groups, key)})
            if binary:
                if g[ycol].nunique() != 2:
                    continue
                pred = g[predcol] if predcol else (g[pcol] >= threshold).astype(int)
                value = classification(g[ycol], g[pcol], pred)
            else:
                value = regression(g[ycol], g[pcol])
            metrics.append(dict(**info, **value))
    for endpoint, prefix in [('classification', ''), ('tg', 'TG/')]:
        original = read(prefix + 'original.csv')
        blind = read(prefix + 'blind.csv')
        labels = read(prefix + 'blind_with_activity.csv')
        check(endpoint + '_unique_labels', original.Code.is_unique and (not original.Code.isna().any()), n=len(original))
        check(endpoint + '_blind_labels_complete', set(blind.Code) == set(labels.Code) and labels.Code.is_unique, n=len(blind))
        check(endpoint + '_train_blind_ids_disjoint', not set(original.Code) & set(blind.Code))
        observations[endpoint + '_target'] = {'n': len(original), 'min': float(original.activity.min()), 'max': float(original.activity.max())}
    split = read('Final_clean_split_seed42.csv')
    check('classification_split_counts', split.split.value_counts().to_dict() == {'Train': 511, 'Internal_evaluation': 128}, counts=split.split.value_counts().to_dict())
    tg_split = read('TG/TG_clean_split_seed42.csv')
    mask = tg_split.split.str.lower().isin(['train', 'outer_train', 'training'])
    check('tg_split_counts', mask.sum() == 208 and (~mask).sum() == 55, n_train=int(mask.sum()), n_test=int((~mask).sum()))
    gcol = 'Core_NADES_ID'
    check('tg_parent_groups_disjoint', not set(tg_split.loc[mask, gcol]) & set(tg_split.loc[~mask, gcol]), train_groups=tg_split.loc[mask, gcol].nunique(), test_groups=tg_split.loc[~mask, gcol].nunique())
    for name, prefix, features_path in [('classification', '', 'DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv'), ('tg', 'TG/', 'TG/TG_RF_DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv')]:
        features = read(features_path)['feature'].tolist()
        check(name + '_final60_unique', len(features) == 60 and len(set(features)) == 60)
        check(name + '_metadata_not_features', not set(features) & {'Code', 'activity', 'Core_NADES_ID', 'A_id', 'B_id', 'C_id', 'D_id'})
        check(name + '_features_in_master', set(features) <= set(read(prefix + 'master_train_features.csv', nrows=0).columns))
    for name, path in [('classification', 'master_train_features.csv'), ('tg_parent', 'TG/master_train_features1.csv')]:
        cols = read(path, nrows=0).columns
        ratios = [c for c in cols if c.startswith('AB_ratio_')]
        use = ['Code'] + ratios + ['A_' + c[9:] for c in ratios] + ['B_' + c[9:] for c in ratios]
        d = read(path, usecols=use)
        direct_bad = absolute_bad = finite_count = 0
        for c in ratios:
            base = c[9:]
            a = pd.to_numeric(d['A_' + base], errors='coerce').to_numpy()
            b = pd.to_numeric(d['B_' + base], errors='coerce').to_numpy()
            actual = pd.to_numeric(d[c], errors='coerce').to_numpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                direct = a / (b + 1e-06)
                absolute = np.abs(a) / (np.abs(b) + 1e-06)
            finite = np.isfinite(actual) & np.isfinite(direct) & np.isfinite(absolute)
            finite_count += int(finite.sum())
            direct_bad += int((~np.isclose(actual[finite], direct[finite], rtol=1e-07, atol=1e-08)).sum())
            absolute_bad += int((~np.isclose(actual[finite], absolute[finite], rtol=1e-07, atol=1e-08)).sum())
        observations[name + '_ratio'] = {'n_ratio_columns': len(ratios), 'finite_comparisons': finite_count, 'directed_mismatches': direct_bad, 'absolute_mismatches': absolute_bad, 'epsilon': 1e-06}
        check(name + '_ratio_matches_directed', direct_bad == 0, **observations[name + '_ratio'])
    for name in ['train', 'blind']:
        p = f'TG/master_{name}_features1.csv'
        s = f'TG/master_{name}_features.csv'
        parent = read(p).set_index('Code')
        solution = read(s).set_index('Code')
        water = read('TG/' + ('original' if name == 'train' else 'blind') + '_water_frc.csv').set_index('Code').water_frc
        check('tg_' + name + '_water_range', water.between(0, 1, inclusive='left').all())
        check('tg_' + name + '_scaling_codes', set(parent.index) == set(solution.index) == set(water.index))
        solution = solution.reindex(parent.index)
        factor = 1 - water.reindex(parent.index)
        features = [c for c in parent.columns if c not in ['activity', 'Core_NADES_ID']]
        expected = parent[features].to_numpy(float) * factor.to_numpy()[:, None]
        actual = solution[features].to_numpy(float)
        match = np.isclose(actual, expected, rtol=1e-07, atol=1e-08, equal_nan=True)
        check('tg_' + name + '_all_column_scaling', match.all(), n_features=len(features), n_entries=int(match.size), mismatches=int((~match).sum()))
        if name == 'train':
            check('tg_target_unchanged', np.array_equal(parent.activity.to_numpy(), solution.activity.to_numpy()))
    pred_metrics('Final60_Internal_Evaluation/Final60_internal_evaluation_predictions.csv', 'activity', 'probability_NADES', binary=True, predcol='predicted_activity')
    blind = read('Final60_blind_predictions.csv').merge(read('blind_with_activity.csv')[['Code', 'activity']], on='Code', validate='one_to_one')
    metrics.append(dict(source='Final60_blind_predictions.csv + blind_with_activity.csv', **classification(blind.activity, blind.probability_NADES, blind.predicted_activity)))
    pred_metrics('TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_internal_evaluation_predictions.csv', 'experimental_Tg_K', 'predicted_Tg_K')
    pred_metrics('TG/TG_Final60_RF_External_Blind_Evaluation/TG_Final60_external_blind_predictions.csv', 'experimental_Tg_K', 'predicted_Tg_K')
    for p, g in [('Final60_LOCO/Final60_LOCO_sample_predictions.csv', 'held_out_component'), ('NestedLOCO_Final60/NestedLOCO_sample_predictions.csv', 'held_out_component')]:
        pred_metrics(p, 'true_activity', 'probability_NADES', groups=[g], binary=True, predcol='predicted_activity')
    pred_metrics('TG/TG_Final60_RF_LOCO/TG_Final60_LOCO_sample_predictions.csv', 'experimental_Tg_K', 'predicted_Tg_K', groups=['component_held_out'])
    cross = read('CrossTask_Family_Multisplit_RF/per_fold_predictions.csv')
    tg = cross[cross.task == 'tg']
    saved = read('CrossTask_Family_Multisplit_RF/tg_logo_pooled_metrics.csv')
    for (variant, family), g in tg.groupby(['variant', 'family']):
        m = regression(g.y_true, g.prediction_Tg_K)
        metrics.append(dict(source='CrossTask_Family_Multisplit_RF/per_fold_predictions.csv', variant=variant, family=family, **m))
        ref = saved[(saved.variant == variant) & (saved.family == family)]
        check('crossfit_tg_' + variant + '_' + family, len(ref) == 1 and np.isclose(m['r2'], ref.r2.iloc[0], rtol=1e-10, atol=1e-10), r2=m['r2'], n_unique_codes=g.Code.nunique(), n_rows=len(g))
    original = read('original.csv').set_index('Code')
    nested = read('NestedLOCO_Final60/NestedLOCO_sample_predictions.csv')
    component_cols = ['A_id', 'B_id', 'C_id', 'D_id']
    for component, g in nested.groupby('held_out_component'):
        members = original[component_cols].eq(component).any(axis=1)
        check('nested_test_members_' + component, set(g.Code) == set(original.index[members]) and g.Code.is_unique, n_test=len(g))
    selected = read('NestedLOCO_Final60/NestedLOCO_fold_selected_features.csv')
    observations['nested_selected_features_columns'] = selected.columns.tolist()
    observations['nested_prediction_folds'] = nested.held_out_component.nunique()
    observations['ab_swap_saved_outputs'] = [str(p.relative_to(root)) for p in root.rglob('*.csv') if any((t in p.name.lower() for t in ['swap', 'ab_position', 'position_swap']))]
    observations['original_environment_lock_files'] = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name.lower() in ['requirements.txt', 'environment.yml', 'environment.yaml', 'pip-freeze.txt', 'pyproject.toml']]
    if args.refit:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from xgboost import XGBClassifier
        for endpoint, prefix, features_path, split_path in [('classification', '', 'DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv', 'Final_clean_split_seed42.csv'), ('tg', 'TG/', 'TG/TG_RF_DiversityGA_SubsetSize_Comparison/subset_60/best_selected_descriptors.csv', 'TG/TG_clean_split_seed42.csv')]:
            features = read(features_path).feature.tolist()
            data = read(prefix + 'master_train_features.csv', usecols=['Code', 'activity'] + features)
            splits = read(split_path).set_index('Code').split
            train = data.Code.map(splits).str.lower().isin(['train', 'outer_train', 'training']).to_numpy()
            X = data[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
            y = data.activity.to_numpy()
            Xblind = read(prefix + 'master_blind_features.csv', usecols=['Code'] + features)
            for purpose in ['heldout', 'deployment_blind']:
                fit = train if purpose == 'heldout' else np.ones(len(data), dtype=bool)
                imp = SimpleImputer(strategy='median')
                xf = imp.fit_transform(X.loc[fit])
                xt = imp.transform(X.loc[~train] if purpose == 'heldout' else Xblind[features])
                if endpoint == 'classification':
                    xf = xf.astype(np.float32)
                    xt = xt.astype(np.float32)
                    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=82, eval_metric='logloss', tree_method='hist', n_jobs=2, scale_pos_weight=float((y[fit] == 0).sum() / (y[fit] == 1).sum()))
                else:
                    model = RandomForestRegressor(max_depth=6, min_samples_leaf=1, min_samples_split=2, n_estimators=180, random_state=42, n_jobs=2)
                model.fit(xf, y[fit])
                prediction = model.predict_proba(xt)[:, 1] if endpoint == 'classification' else model.predict(xt)
                path = ('Final60_Internal_Evaluation/Final60_internal_evaluation_predictions.csv' if purpose == 'heldout' else 'Final60_blind_predictions.csv') if endpoint == 'classification' else 'TG/TG_Final60_RF_Internal_Evaluation/TG_Final60_internal_evaluation_predictions.csv' if purpose == 'heldout' else 'TG/TG_Final60_RF_External_Blind_Evaluation/TG_Final60_external_blind_predictions.csv'
                pcol = 'probability_NADES' if endpoint == 'classification' else 'predicted_Tg_K'
                current = pd.DataFrame({'Code': data.loc[~train, 'Code'].to_numpy() if purpose == 'heldout' else Xblind.Code.to_numpy(), 'refit_prediction': prediction})
                comparison = current.merge(read(path)[['Code', pcol]], on='Code', validate='one_to_one')
                delta = np.abs(comparison.refit_prediction - comparison[pcol])
                observations[endpoint + '_' + purpose + '_refit'] = {'n': len(comparison), 'max_abs_prediction_difference': float(delta.max()), 'mean_abs_prediction_difference': float(delta.mean()), 'allclose_rtol1e-6_atol1e-6': bool(np.allclose(comparison.refit_prediction, comparison[pcol], rtol=1e-06, atol=1e-06)), 'threads': 2}
                comparison.to_csv(out / (endpoint + '_' + purpose + '_refit_predictions.csv'), index=False)
    env = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': {}}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'joblib', 'statsmodels']:
        try:
            env['packages'][package] = version(package)
        except PackageNotFoundError:
            env['packages'][package] = None
    result = {'scope': 'Archive integrity, encoding consistency and saved-prediction metrics; optional fixed-feature refits only. No full pipeline reproduction.', 'checks': checks, 'observations': observations, 'audit_environment_not_original': env}
    (out / 'deposit_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    pd.DataFrame(metrics).to_csv(out / 'recomputed_metrics.csv', index=False)
    print(json.dumps({'checks': len(checks), 'passed': sum((c['passed'] for c in checks)), 'failed': [c['name'] for c in checks if not c['passed']], 'observations': observations}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
