# Submission library assembly audit

Run: `final_ar_evo_20260928`

This submission entry uses the conservative autoregressive/evolutionary composition. Any
candidate with HydrAMP or AMP-Diffusion ancestry was excluded. The library contains
3,964 unique, scored AR/evolution candidates
followed by 46,036 unique ProGen2 auxiliary sequences.
The auxiliary sequences were selected from 122,234
unique non-reference attempts after the shared AMP/safety model and synthesis filter.

| Check | Result |
| --- | ---: |
| Library sequences | 50,000 |
| Unique library sequences | 50,000 |
| Synthesis-filter passes | 50,000 |
| Exact reference overlaps | 0 |
| Unique Top 100 | 100 |
| Top 100 subset of library | True |
| Top-100 max similarity to references | 0.800000 |
| Top-100 maximum internal similarity | 0.777778 |
| Reserve rows | 200 |

The 50,000-row library is screened with the project synthesis filter. Its AMP
probabilities and toxicity estimates are model predictions; auxiliary APEX MIC is not
available and no output is a wet-lab assay result. Source attribution and input hashes are
recorded in `library_assembly_manifest.json` and `submission_entry_manifest.json`.
