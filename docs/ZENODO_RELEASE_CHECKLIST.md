# Zenodo Release Checklist

## Software record

- Upload source from the tagged GitHub release.
- Title: `Offline Safe RL Risk Audit`
- Version: `1.0.0`
- License: Apache-2.0
- Related identifier: the replication-data DOI
- Resource type: Software

## Data record

- Upload `offline-safe-rl-risk-audit-replication-data-v1.0.0.zip`.
- Title: `Replication Data for Offline Safe RL Risk Audit`
- Version: `1.0.0`
- License: CC BY 4.0
- Resource type: Dataset
- Related identifier: the software DOI and the article DOI

## Required metadata

- Final author names, order, affiliations, and ORCID iDs.
- Final article title.
- GitHub repository URL.
- Software DOI and dataset DOI.
- Funding information, if any.

## Post-publication checks

1. Download the published archive once and compare its SHA256 with the local
   archive.
2. Open the DOI links in a logged-out browser.
3. Confirm the `datasets/` and `manifests/SHA256SUMS.txt` paths are visible.
4. Add both DOI links to the manuscript's Data Availability Statement.
5. Replace the generic author entity in both `CITATION.cff` files.
