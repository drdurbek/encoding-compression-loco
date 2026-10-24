# Original figure-generation sources

The plotting code, styling and output basenames are retained from the supplied
scripts. No new plot design or replacement numerical analysis is introduced.
Use a working copy of Curated for actual runs; some original scripts fit models
and replace result exports before plotting.

All commands below are run from the code root. Replace ../work/Curated with
your working data folder. Appending --dry-run checks compilation only.

| Figure / source | IDs to pass after --analysis | Execution scope |
|---|---|---|
| Main 1–2 conceptual schemes | Original drawing source not supplied | No artwork/code invented |
| Main 3 classification | classification_06 classification_08 classification_09 classification_10 | Fitting + plots; one command required |
| Main 3 Tg | tg_08 | RF fitting + parity |
| Main 4 classification | classification_12 | Reads saved LOCO CSVs |
| Main 4 Tg | tg_12 | Fixed-feature LOCO fitting + plots |
| Main 5 | classification_28 classification_29 | Reads saved nested/selected-feature tables |
| Main 6 blind probability | classification_15 | Reads saved blind predictions |
| Main 6 family probe | classification_16 classification_17 | Refits family models + plots; one command required |
| Main 7 family OOF | tg_10 | Family OOF fitting + plots |
| Main 7 concentration blind | tg_13 | Deployment RF fitting + parity |
| Supplementary 1 | classification_04 tg_06 | Reads saved sweep CSVs; Tg plot also writes its original derived summary |
| Supplementary 2 | classification_05 tg_07 | Fits original PCA/UMAP geometry + plots |
| Common K=8 comparison | classification_24 | Cross-fitting of both endpoints + original plot |
| Other learner/family plots | classification_07 classification_18 classification_25 tg_11 tg_14 tg_16 | Fitting + their original plots |

Examples of saved-result plotting:

```bash
python tools/run_analysis.py --analysis classification_12 classification_15 classification_28 classification_29 --data-root "../work/Curated"
python tools/run_analysis.py --analysis classification_04 tg_06 --data-root "../work/Curated"
```

Existing component plots are not automatically composed into new manuscript
panels. Original image filenames can differ from final manuscript figure
numbers. The retained Final40 confusion-matrix basename is a historical name
for a plot made from current Final60 variables; no filename was silently changed.
The historical 639-row Final40_CV exports are distinct from the 511-row
Final60_TrainOnly_CV analysis used for the relevant development curves.

The separate Figures_Code.zip contains 24 plot-capable sources, this runner,
path resolver and file map. Non-plot prerequisites such as initial encoding or
nested fitting are in the complete GitHub_Code.zip. Prepare those inputs first
if you choose to recompute them rather than use saved outputs.

Path resolution/compilation were checked. Figures were not rendered during
this update. Fresh-classification fit differences previously recorded in the
audit remain unresolved and may affect newly fitted curves.
