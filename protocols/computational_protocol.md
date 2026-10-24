# Encoding compression and component withholding validation of multicomponent materials

## Scope

Computational protocol draft for the NADES classification and Tg regression case studies. The study parameters below follow the supplied manuscript and analysis transcripts. A final protocol must identify the canonical source scripts and actual archived data/software versions before publication.

## Before starting

Obtain the files specified in `data/DATA_REQUIREMENTS.md`, the verified Python environment and the original descriptor-generation software versions/settings. Use the data-verified directed ratio A/(B+1e-6), and align the manuscript equation. Confirm the sample ID and component registries. Read the execution index to identify state-dependent cells and historical variants. Timing, RAM and hardware requirements remain to be measured on the canonical run.

## Procedure

1. **Prepare endpoint-specific inputs.** Maintain separate classification and Tg datasets. Verify 639 labelled classification samples and the 34 blind formulations against the frozen curation manifest. For Tg, verify 263 labelled measurements, 33 parent systems and 20 blind concentration samples. Retain the 51 external DES measurements as a separately identified domain-diagnostic dataset. Keep sources and exclusion reasons.
2. **Prepare component structures and descriptors.** Retain the exact structures, protonation/salt conventions, component IDs, slot ordering, geometry optimization settings and molecular-descriptor export used in the study. The manuscript specifies ChemSketch, HyperChem 8/RM1 and alvaDesc; exact versions/options and source structure files are needed. Export Code, val1–val4 as appropriate and descriptor columns to the component tables.
3. **Construct mixture representations.** Generate component-resolved, arithmetic-mean, stoichiometry-weighted, pairwise A–B and auxiliary stoichiometric features with the confirmed operator definitions. The supplied master matrices use the directed A/(B+1e-6) ratio; do not replace it with an absolute-value ratio. Missing constituents and missing descriptor values must be handled according to the frozen code. Save the column definitions and checksums of the master matrices. Do not alter the ratio formula to match the manuscript without tracing the original result source.
4. **Apply Tg concentration scaling.** Join water_frc by Code. Apply \(x_{ij}^{\mathrm{solution}}=(1-w_i)x_{ij}^{\mathrm{parent}}\) for every encoded predictor \(j\), where \(w_i\) is `water_frc`. Retain identifiers and experimental Tg unchanged. Document the experimental basis of water_frc and check its valid range. The supplied code scales auxiliary stoichiometric predictors too. Preserve a copy of the unscaled parent matrix.
5. **Define parent groups and fixed partitions.** Create the order-invariant Tg parent identifier from components and normalized stoichiometry, while retaining the encoding's component slots. Save exact split membership: classification stratified 511/128; Tg grouped 208/55 with 26/7 parents. The provided scripts use seed 42 for these splits. A saved membership file is authoritative for a final run.
6. **Perform descriptor filtering and ranking.** Document the precise fit scope of each label-free QC stage. On development data, compute endpoint-appropriate association, discrimination, mutual information and BH-adjusted significance scores. Use the confirmed inclusive-OR eligibility rules and percentile-ranking weights; shortlist 4,000 descriptors and apply correlation pruning at absolute r = 0.95. Preserve retained/rejected lists and reasons. Main candidate pools are 1,000 classification and 560 Tg descriptors; explain any run-specific deviations.
7. **Perform model-guided selection.** For each candidate subset, fit the imputer and learner within inner training folds. Use stratified folds for classification and parent-grouped folds for Tg. Run the documented 10–60 size sweep and independent GA searches. Keep all seeds, settings, descriptor lists, OOF predictions and fitness histories. Select from inner-development results and freeze the chosen Final60 set. Do not select from held-out or blind performance.
8. **Evaluate the held-out partitions once.** Use the frozen model settings, feature set and classification threshold. Save per-sample predictions, metric definitions, bootstrap resampling units and uncertainty outputs. Distinguish this evaluation model from the later deployment refit. The manuscript's classification threshold is 0.42; verify it against the saved OOF selection record.
9. **Run fixed-representation LOCO.** Withhold every formulation containing a component, refit the model with the fixed Final60 representation and document that descriptor identity was selected earlier. Save held-out component IDs, predictions and per-component eligibility flags. Do not label this as fully nested feature selection.
10. **Run fully nested classification LOCO.** Within each outer component exclusion, repeat technical cleaning, supervised screening, ranking, correlation pruning, DAGA selection and inner-OOF threshold selection using the remaining formulations. Keep 60 descriptors per outer fold while reselecting their identities. Save fold-specific selected columns, thresholds, seeds, group membership and metrics. Distinguish the 31 evaluable mixed-class components from all 124 component candidates.
11. **Run equal-budget family cross-fitting.** Repeat the full train-only family selection at K = 8 descriptors per family. The provided analysis uses repeated stratified five-fold classification splits and leave-one-parent-out Tg splits. Retain common partitions across families. Keep this analysis separate from post-selection decomposition of Final60.
12. **Refit deployment models and predict blind inputs.** After final selection and held-out evaluation, refit the frozen configuration on all labelled endpoint data. Generate blind predictions before joining blind labels. Preserve prediction files and hashes. Evaluate classification blind labels separately from Tg concentration extrapolation, and identify the exact source of each family-only analysis.
13. **Perform external-domain diagnostics.** Assess the external DES dataset with the confirmed baselines, then label the anhydrous-subset analysis as post-hoc. Do not treat a dataset inspected during development as an untouched confirmatory validation set.
14. **Generate source data and manuscript outputs.** Export numerical data underlying every figure, including OOF/held-out curves, confusion matrices, LOCO tables, family comparisons, GA size sweeps and UMAP coordinates. Save visualization fit scope, parameters and seeds. UMAP is descriptive, not a feature-selection criterion.
15. **Verify and archive.** Reproduce the reported outputs from a fresh session, document acceptable numerical tolerances and disclose any remaining differences. Measure runtime and peak memory. Finalize code/data licenses and contributor metadata. Archive the exact release, link dataset and protocol records, and insert their real DOIs into the manuscript.

## Troubleshooting

- A missing Code or inconsistent component identifier stops joins and invalidates grouping: correct the curation manifest rather than dropping the sample silently.
- An undefined variable in an extracted script indicates a session dependency: identify the prerequisite analysis or turn the cell into a self-contained function before release.
- A difference in descriptor counts may reflect all-data versus train-only cleaning: compare the fit scope and saved feature lists before choosing a count.
- A changed family ranking may reflect a different split, feature budget or analysis stage: compare the saved run manifest, not only an aggregate metric.
- Failure to match a published result must be documented and resolved before claiming reproducibility.

## Verified archived-model reproduction

For the supplied data, install `requirements-model-check.txt` and run the four `tools/predict_archived_models.py` examples in README.md. Their predictions match archived exports within 1e-6. `tools/verify_deposit.py --refit` reproduces fixed-feature Tg fits but does not currently reproduce classification refits. Selection, nested LOCO, uncertainty and all figures have not been rerun. The draft must not claim that the entire protocol is validated until these gaps are closed.

## Records to attach

Final data and software DOIs; exact release tag and commit; input/output checksums; environment lock; sample/component dictionaries; split files; Final60 selections; nested selections; source-data tables; measured runtime/hardware; experimental protocol links where applicable.

## Component-slot sensitivity

The author confirmed that no A/B position-swap control was performed. Slot-assignment sensitivity remains an untested limitation. The supplied candidate swap script is retained as optional, unexecuted source and must not be cited as a completed result.


## Current package paths

Use the original-named files in the manually assembled Curated layout.
Pass the Curated folder to tools/run_analysis.py --data-root. See README.md,
metadata/FILE_MAP.csv and docs/FIGURES.md for exact locations and execution
prerequisites. No copying/renaming adapter or flat legacy project tree is needed.
The standalone superseded blocks are excluded from the current launcher.
