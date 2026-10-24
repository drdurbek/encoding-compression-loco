> Historical scientific audit, retained for scope/provenance. Current path
> configuration and checks are in README.md and metadata/path_update_validation.json.
> The old execution layout and removed source blocks are not the current launcher.

# Release audit — 22 October 2026

## Supplied and checked

The authors supplied `07092026New.rar`: inputs, master matrices, splits, GA selections, trained models and saved predictions. Exact member checksums are in `metadata/data_file_inventory.csv`. No standalone Python or notebook files were found in that archive; analysis sources still come from the previously supplied transcripts.

`tools/verify_deposit.py` passed 61 explicit data-consistency checks. The 31 saved nested-LOCO test folds match original component membership. Main split counts and Tg parent disjointness match the paper. All 56,682 scaled Tg predictors were checked for both train and blind matrices, including auxiliary stoichiometry. Numeric metric comparisons are recorded separately; passing integrity checks is not full pipeline reproduction.

## Confirmed ratio definition: fix the Methods equation

Across 5,667 ratio columns in each parent master matrix, all finite comparisons match `A/(B+1e-6)`. The alternative `abs(A)/(abs(B)+1e-6)` differs in 172,694 classification entries and 67,193 Tg-parent entries. These are counts of matrix entries, not independent observations. Keep the code/data operator and align the Methods formula, unless the authors deliberately regenerate the complete study using another operator.

## Archived models versus new training

All four archived-model prediction checks match the saved exports within 1e-6 tolerance. Fixed-feature Tg RF refits match to about 1e-13 K. Fresh classification fits differ: maximum held-out probability difference about 0.189 and blind difference about 0.132 with XGBoost 2.1.3/scikit-learn 1.5.2; changing CPU thread counts did not remove the held-out difference. A newer XGBoost also differed. Version matching alone did not resolve this. The cause could involve unrecorded environment, input or training details and is not established here.

The stored XGBoost JSON records version 2.1.3; serialized sklearn estimators record 1.5.2. The complete original Python/NumPy/SciPy environment, input hashes at fit time and original cold-start training run are absent. `requirements-model-check.txt` reproduces the audit/prediction environment, not original development. Resolve or transparently delimit retraining reproducibility before claiming end-to-end reproduction.

## Confirmed cross-fitted family results

`CrossTask_Family_Multisplit_RF/tg_logo_pooled_metrics.csv`, independently recalculated from its per-fold predictions, reports weighted R² 0.548408, arithmetic mean 0.407152, and component/pairwise families 0.130255–0.168707. These belong to the common K=8 family cross-fitting analysis, distinct from the Final60 family decomposition and blind tests.

## Other remaining issues

- The main technical prefilter uses full-matrix label-free QC, then train-only supervised ranking. Nested LOCO repeats QC within each outer training fold. Match stage-specific Methods claims to that source; do not silently replace code and retain old results.
- The author confirmed on 22 October 2026 that the A/B position-swap control was not performed. The earlier transcript includes an unexecuted candidate script. Treat slot-assignment sensitivity as an untested limitation, not a completed analysis or missing mandatory result.
- Historical Final40/Final50 classification and Final10 Tg artifacts coexist with Final60; the deposit index labels them. Some Final40-named plotting files sit inside the Final60 output folder and must not be selected by directory name alone.
- Two classification plotting/calibration source cells rely on session state. See the execution index. The complete nested/DAGA workflow has not been rerun here.
- Historical structure files are supplied separately without an authoritative full ID-to-structure registry. Confirm complete component coverage, slot rules, descriptor-generation settings and provenance before claiming descriptor regeneration from structures.
- `water_frc` is verified numerically, but its physical fraction basis must be stated by the authors. Experimental preparation/measurement settings are not inferred from files or labels.
- The classification blind summary reports zero labelled rows despite a complete separate blind label file. This is a stale/unjoined export; the audit recomputes joined blind metrics in a separate output. Keep the original export as provenance.
- Workbook labels such as `Table S1` are retained verbatim in raw evidence and exported CSVs; they do not define the final manuscript's Supplementary Table numbering. Final SI numbers must be mapped to the final table contents.
- Finalize contributor roles, licenses, source-data figure assignments and actual DOI/repository links. No record has been publicly deposited by this preparation.

## Scope

This is a documented review build with verified archived prediction and numerical consistency, not a claim that every result, confidence interval, figure or selection run has been recreated. All unresolved issues are retained explicitly.

All six master matrices were regenerated with original encoding scripts in an isolated project. Numeric values, column order, sample IDs and Tg parent IDs match the author archive (rtol 1e-7, atol 1e-8; zero mismatches).
