# GitHub Zenodo and protocols.io

## GitHub

Use a descriptive repository name such as `nades-encoding-compression-loco`. Assemble the canonical scripts, verified environment, small reproducible example, complete execution instructions and a mapping from analysis steps to final outputs. Add the selected software license and finalize citation metadata. Run the workflow in a fresh environment. Capture exact versions and checksums. Record a commit hash and create a versioned release only after the results are checked.

## Zenodo

Use Zenodo as the persistent DOI archive. Either connect the chosen GitHub repository before a new release or upload the verified software release manually. A separate dataset record is useful when the study matrices and source data need their own metadata or are too large for the code repository. Link code, dataset, protocol and paper records with their real identifiers. Cite the exact archived software version used for the paper.

Zenodo can reserve a DOI in a draft record. A reserved DOI is not evidence that the record has been published or is publicly accessible. Do not insert an invented DOI in the manuscript. Zenodo supports CITATION.cff and .zenodo.json; if both exist, .zenodo.json takes precedence for GitHub release metadata. This preparation package provides templates rather than active release metadata.

## protocols.io

Use `protocols/computational_protocol.md` as the starting text for a computational protocol. Replace missing file paths, confirmed versions, elapsed times and final repository/DOI links after verification. Add screenshots or attachments only if they help execution. Keep the actual protocol version synchronized with the archived software release. A separate laboratory protocol can document mixture preparation and experimental classification/Tg measurement if those experiments are part of the paper; the complete experimental settings still require author confirmation.

Nature Portfolio generally encourages protocols sharing. The central code and data must meet the selected journal's requirements and be accessible for review as required. A journal-specific editorial instruction to use protocols.io takes precedence over the general recommendation; retain that instruction with the submission checklist.

## Sources checked 2026-10-21

- https://www.nature.com/ncomms/submit/policy
- https://www.nature.com/ncomms/editorial-policies/reporting-standards
- https://www.nature.com/nature/for-authors/formatting-guide
- https://help.zenodo.org/docs/github/
- https://help.zenodo.org/docs/github/describe-software/
- https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/

## Files ready for review

- `GitHub_Code.zip`: unpack into a new repository. Keep the Curated data folder out of git.
- `Zenodo_Curated.zip`: upload as the accompanying dataset after licenses/provenance are finalized. Retain the existing Curated layout and pass its location with --data-root.
- `protocols_io_draft.md`: protocol text for transfer into protocols.io steps; retain its draft status until the documented reproduction gaps are resolved.

Before the first public software release, finalize CITATION.cff from its template, add the agreed LICENSE, and record release tag, commit and actual archive identifiers. Use the archived version DOI in the paper. Create linked software and dataset records rather than putting the large data ZIP into git. Uploading review files and public DOI publication are separate steps.
