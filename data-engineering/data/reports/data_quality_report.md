# Data quality report

- Source-standardized records: 122,581
- Unique canonical normalized sequences: 38,678
- Valid 8–50 aa, standard-alphabet, no-known-modification sequences: 27,509
- Eligible generation corpus (also excludes exact challenge-reference matches): 4,051
- Challenge reference sequences: 39,448; exact overlap flagged: 23,590
- Source-record-to-canonical surplus (repeated assays and sequence duplicates): 83,903
- Activity observations: 20,491; toxicity observations: 1,609

## Similarity and leakage

The challenge-safe generation corpus is exhaustively compared pairwise using RapidFuzz's Indel similarity, which matches the official validator's `Levenshtein.ratio` semantics. Connected components are written for 80%, 70%, and 60% thresholds. Train/validation/test assignment is deterministic (seed 42) and keeps every 80%-similarity component in a single split.

## APD3 and AMPlify

APD3 is not a standalone local source. AMPlify is preserved as a consolidated source and quantified in `overlap_matrix.csv`; it is not counted as independent biological evidence.
