# Data requirements

Use --data-root to select Curated. See metadata/FILE_MAP.csv for all 312 canonical
files. tools/run_analysis.py --check-data validates their presence; --check-hashes
also checks the supplied frozen snapshot. Data files are never renamed.

| Data | Actual path |
|---|---|
| 639 classification labels | classification/input/original.csv |
| 34 blind identities / labels | classification/input/blind.csv; blind_with_activity.csv |
| Component tables | endpoint/input/a.csv, b.csv, c.csv, d.csv |
| 263 Tg targets | regression/input/original.csv |
| 20 concentration identities / labels | regression/input/blind.csv; blind_with_activity.csv |
| Tg water fractions | regression/input/original_water_frc.csv; blind_water_frc.csv |
| Classification master matrices | classification/encoded/ |
| Tg parent/scaled master matrices | regression/encoded/ |
| Tg parent table | regression/result/TG_core_nades_groups.csv |
| 51 external DES | regression/blind_2/ |
| Workbooks | classification/input/Table1-3.xlsx; regression/input/Table1-3.xlsx |

Endpoint Codes are local identifiers. activity is binary for classification
and Tg in kelvin for regression. No complete verified molecular structure
collection is supplied; descriptor-generation provenance remains separate.
