# Data package

Use the existing Zenodo_Curated/Curated directory and pass it with --data-root.
The directories classification/input, classification/encoded,
classification/result, classification/blind and their regression counterparts
are used directly. The external data are in regression/blind_2. No data-name
conversion or project-tree assembly is required. metadata/FILE_MAP.csv records
the exact original-to-package paths. Do not put the large data archive into git.
