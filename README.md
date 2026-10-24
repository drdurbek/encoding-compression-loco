# Encoding, compression and component-withholding validation

Author analysis code adapted to the manually assembled NADES `Curated` package.
The archive contains 45 retained analysis blocks with their existing filenames.
Changes are limited to path configuration, execution instructions and four stale
diagnostic messages. Scientific equations, parameters, selection rules, seeds,
classification thresholds and plot styles are retained.

## Authors, citation and license

Software authors, in order: Durbek Usmanov, Allison Hubel, and Bakhtiyor Rasulev.
Their institutional affiliations and software citation metadata are recorded in
`CITATION.cff`. Version `1.0.0` is the version prepared for the first software
release; its DOI will be added after Zenodo issues it.

The author-developed source code and accompanying documentation are distributed
under the MIT License; see `LICENSE`. Third-party dependencies retain their own
licenses. This software license does not assign new reuse terms to third-party
data or to the separately deposited Curated dataset.

## Data location and names

Pass your existing `Curated` folder to `--data-root`. The package reads:

| Contents | Actual package location |
|---|---|
| Classification metadata and component descriptors | `classification/input/original.csv`, `blind.csv`, `blind_with_activity.csv`, `a.csv`–`d.csv` |
| Classification encoded matrices | `classification/encoded/master_train_features.csv`, `master_blind_features.csv` |
| Classification filtering, evaluation, LOCO and selection | `classification/result/` |
| Classification blind and Component B readouts | `classification/blind/` |
| Tg metadata, component descriptors and water fractions | `regression/input/` |
| Tg parent and scaled matrices | `regression/encoded/` |
| Tg grouped evaluation, LOCO and selection | `regression/result/` |
| Tg blind concentration/family analyses | `regression/blind/` |
| External DES inputs and analyses | `regression/blind_2/` |
| Common K=8 family cross-fitting | `CrossTask_Family_Multisplit_RF/` |

Both author-renamed workbooks remain `Table1-3.xlsx` in their endpoint input
folders. `metadata/FILE_MAP.csv` maps original project paths to the actual
package paths. `tools/project_paths.py` uses this map directly; it does not
rename, copy, assemble or convert any data file. New original-named analysis
outputs are routed to the corresponding result/blind/encoded directories.

## First check: no analysis execution

Use Python 3.12. From this code folder, run:

```bash
python tools/run_analysis.py --check-data --check-hashes --data-root "../Zenodo_Curated/Curated"
```

Replace the example path with your actual `Curated` location. The command checks
the 312 canonical files and their SHA-256 values. It does not train models,
generate predictions, render figures or write analysis results. Extra README
files are permitted. Hash comparisons describe the supplied archived snapshot;
rerunning analyses may intentionally replace outputs and change their hashes.

```bash
python tools/run_analysis.py --list
python tools/run_analysis.py --analysis classification_01 tg_01 tg_02 tg_03 --data-root "../Zenodo_Curated/Curated" --dry-run
```

`--dry-run` compiles the requested source only. It does not establish successful
model fitting or figure rendering.

## Environment and actual execution

`requirements-model-check.txt` records the previously tested archived-model
environment. `requirements.in` lists dependencies used by the full source;
it is not an exact lock of the original development environment.

```bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements-model-check.txt
# For full analyses/plots, also install the missing imports from requirements.in
# in a recorded environment.
```

For actual analysis, first make a working copy of `Curated`. Original scripts
write into their data root and can overwrite archived outputs. Pass the working
copy to `--data-root`. There is no implicit prerequisite fitting: dependent
notebook cells require their prerequisite IDs explicitly earlier in the same
command. The runner checks this before starting.

See `docs/EXECUTION_ORDER.md` for stages and `docs/FIGURES.md` for plot commands.
For standalone source execution, set the environment variable `NADES_DATA_ROOT`
to the Curated root; state-dependent cells still require prior in-memory state.
The preferred entry point is `tools/run_analysis.py`.

## Saved-model tools

These commands run inference/audits only when you explicitly invoke them:

```bash
python tools/predict_archived_models.py --data-root "../Zenodo_Curated/Curated" --endpoint classification --purpose heldout --out "../checks/classification_heldout.csv"
python tools/predict_archived_models.py --data-root "../Zenodo_Curated/Curated" --endpoint classification --purpose deployment --out "../checks/classification_blind.csv"
python tools/predict_archived_models.py --data-root "../Zenodo_Curated/Curated" --endpoint tg --purpose heldout --out "../checks/tg_heldout.csv"
python tools/predict_archived_models.py --data-root "../Zenodo_Curated/Curated" --endpoint tg --purpose deployment --out "../checks/tg_blind.csv"
python tools/verify_deposit.py --data-root "../Zenodo_Curated/Curated" --out "../checks/audit"
```

The optional `verify_deposit.py --refit` retrains fixed-feature models. It does
not repeat screening or DAGA and is not part of a path-only package check.
Historical fresh-classification training differences remain unresolved; path
adaptation does not establish identical new training results.

## Retained and removed sources

- Removed: the standalone older Final40 blind block, the Final40 calibration
  block, and the historical Tg weighted-only comparator whose inputs were
  intentionally excluded from the curated data. Their analysis IDs are not
  accepted by this runner. Remaining script basenames are unchanged.
- Retained: the 10–60 sweep, including 40 and 50 points used by Supplementary
  Figure 1. Historical image basenames in retained scripts are preserved.
- A/B swap remains an optional unexecuted control, not a completed result.
- Complete raw transcripts are not duplicated in this release because they
  include excluded blocks and console output. Source hashes/ranges remain in
  `metadata/source_inventory.json`.

## Verification of this path update

All 45 retained calculation ASTs match the supplied source after normalizing
only path expressions and the stated diagnostic messages. The two numerical
audit/inference tools likewise retain their non-path ASTs. Thirteen routing
tests passed, all stages compile, and the 312 supplied canonical data hashes
match. See `metadata/path_update_validation.json` and `metadata/path_changes.json`.
No model fitting, prediction generation or figure rendering was performed for
this update. This is not full scientific workflow validation.

## Subsequent Windows verification

The archived models were subsequently run on Windows with Python 3.12.10.
All four prediction sets were reproduced: classification held-out (128 rows),
classification blind (34), Tg held-out (55), and Tg blind (20). Classification
probabilities matched the saved predictions exactly; maximum absolute Tg
prediction differences were below 1e-10 K. The deposit audit passed all 61
checks. These checks used saved models and saved validation predictions;
they did not rerun feature selection, DAGA, LOCO or nested LOCO training.

The prediction CSVs, audit report, recomputed metrics and exact installed package
versions are provided in `validation/windows-check/`. The Windows requirements
file records the verified inference/audit environment; additional dependencies
for full training and figure generation are listed in `requirements.in`.

Previously recorded scientific input and retraining questions are not corrected
by changing paths. The incomplete, partly incorrect structure collection is
excluded from the data at the author's direction; ML starts from the supplied
component descriptors. Software authors, affiliations and MIT licensing are
recorded above. Publication identifiers are added only after they have been
issued by their services.
