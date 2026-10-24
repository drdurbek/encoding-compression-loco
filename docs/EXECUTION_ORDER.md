# Execution order

Use the supplied Curated folder directly for --check-data/--dry-run. For actual
execution, use a working copy because original analyses write result files.

| Stage | Classification IDs | Tg IDs |
|---|---|---|
| Encoding | classification_01 | tg_01, tg_02, tg_03 |
| Technical/supervised prefilter | classification_02 | tg_04 |
| DAGA size sweep | classification_03 | tg_05 |
| Sweep plots | classification_04 | tg_06 |
| Geometry/UMAP | classification_05 | tg_07 |
| Final60 evaluation and deployment | classification_06 | tg_08 |
| Model-family benchmark | classification_07 | tg_11 |
| Held-out CM/ROC/PR + training OOF plots | classification_06 classification_08 classification_09 classification_10 | tg_08 |
| Fixed Final60 LOCO | classification_11 then classification_12 | tg_12 |
| Nested classification LOCO | classification_27 then classification_28 / classification_29 | No matched nested Tg selection supplied |
| Frozen blind | classification_14 then classification_15 | tg_13 |
| Final60 family probe | classification_16 classification_17; classification_18 separately | tg_10 |
| Equal-budget common family analysis | classification_24 (both endpoints) | via classification_24 |
| Equal-budget blind families | classification_25 then classification_26 | tg_14 |
| External DES encoding and diagnostics | — | tg_15, tg_16, tg_17, tg_18, tg_19 |

## Session-dependent cells

The runner uses one namespace for the IDs in a single invocation.

- classification_08 / 09 / 10 / 20 / 21 / 22 require classification_06 earlier
  in that same command.
- classification_17 requires classification_16 earlier in the same command.
- No hidden prerequisite fitting is performed. Missing prerequisites are
  rejected before any requested stage starts.
- The two held-out bootstrap output files are explicitly routed to their
  archived Final60_Blind_Family_Probe directory; this removes dependence on
  the incidental output directory left by a notebook session.

Example (executes model fitting and plotting when --dry-run is omitted):

```bash
python tools/run_analysis.py --analysis classification_06 classification_08 classification_09 classification_10 --data-root "../work/Curated" --dry-run
```

Standalone older blocks classification_13, classification_19 and tg_09 are
excluded. The optional classification_23 A/B control was not performed by
the authors; it is not part of the completed-paper execution sequence.

No full scientific rerun was performed during this path update. Source data,
models and numerical settings are unchanged by the adapter.
