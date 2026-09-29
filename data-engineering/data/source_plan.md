# Source plan, version 1.0

| Source | Role | Local status | Inclusion policy |
| --- | --- | --- | --- |
| DBAASP | Peptide provenance, activity and toxicity assays | Local export | Retain all observations; valid sequences may enter core corpus. |
| DRAMP | Broad AMP sequence provenance | Local FASTA/XLSX exports | Canonicalize exact duplicates; retain provenance. |
| dbAMP | Broad AMP sequence provenance | Local CSV export | Canonicalize exact duplicates; retain provenance. |
| AMPlify | Consolidated AMP training export | Local FASTA export | Preserve as provenance; do not count as independent evidence. |
| Challenge antibacterial FASTA | Submission novelty reference | Official downloaded file | Validation/exclusion for novelty-safe candidate views only; not a source of core training rows. |
| APD/APD3 | Not locally supplied | Absent | Not claimed or downloaded. |
| HydrAMP | Optional model resource | Absent | Not merged; consume an exact external release only with separate provenance. |
| Negative controls | Classification support | Absent | Not fabricated from missing labels. |
