# Official reference-match scope

`official_reference_matches.csv` reports exact sequence overlap with the official `antibacterial.fasta` reference and is sufficient for the full-library exact-overlap rule.

Nearest-reference similarity is intentionally not included in dataset version 1.0 because it would require a large all-vs-all reference comparison and is only mandatory for a generated top-100 submission, not the training corpus. The CLI validator applies the official strict top-100 condition when candidate FASTA files are available.
