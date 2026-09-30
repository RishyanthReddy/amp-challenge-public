# Original library replay

Verified 2026-09-29 in isolated workspaces. The submitted FASTAs were preserved.

## Results

| Stage | Observed comparison |
| --- | --- |
| Auxiliary run A | Completed the original 60,000-attempt schedule, seed 200000, batch 64; the first 51,069 rows match the historically interrupted CSV byte-for-byte |
| Auxiliary run B | Completed 85,000 attempts, seed 300000, batch 64; the entire raw CSV matches byte-for-byte |
| Perplexity scoring | Independently scored both retained pools on RTX 4090; both PPL CSV hashes exactly match the originals |
| MAP-Elites/DPP selection | Original Top 100 and nested Top 50 Parquet hashes match |
| Library assembly | The full 50,000-row library Parquet hash matches |
| FASTA export | Both library and ranked Top-100 FASTA hashes match |
| Sequence validation | Counts, uniqueness, alphabet, lengths, subset, headers, reference exclusion and similarity pass |

The rebuilt library contains 3,964 scored candidates and 46,036 auxiliary peptides, all
synthesis-filter passes. Its domain counts are 47,683 autoregressive and 2,317 evolutionary.
Top 100 remains 60 autoregressive and 40 evolutionary candidates. Maximum reference
Levenshtein ratio is 0.8000; maximum internal Top-100 ratio is 0.77777778.

Library SHA-256: `65183eef34ef86924f311c47de9b9183dabad0634da8586d155c01d81944afba`.

Top SHA-256: `7181777317b8a4c7c29841471ae9941dfb0de83b1a0c53c5f0a6bf6d0991e480`.

## Raw-data reconstruction

A fresh source restore verified all 14 inputs, using eight included snapshots and six
pinned downloads. Raw curation recovered the same AR and evaluator views and sequence
table values. Toxicity and provenance differ only in source-path separators; normalizing
backslashes to slashes makes all ordered values equal.

Serialized Parquet hashes differ: the originals record Arrow 23.0.0 and the rebuilt files
record Arrow 25.0.1. The [value fingerprints](curated_view_fingerprints.json) check all
columns, dtypes and ordered scalar values, including exact hexadecimal floats. Only
`source_file` separators are normalized.

Refitting both forests from those rebuilt inputs recovers the original AMP and RBC model
files byte-for-byte. This is stronger evidence than comparing predicted metrics alone.

## Scope

The original auxiliary schedules and final artifacts are now reproduced. Selection and
assembly used the unchanged historical audited/APEX-scored pool. AR training, initial
3,500-candidate sampling, evolution and APEX scoring were verified separately in the
[2026-09-28 replay](FULL_REPLAY_VERIFICATION.md). This was not one uninterrupted training
job starting from raw data. The root command still exports frozen selected tables.

Run A's original cancellation remains part of its provenance. The replay completed its
full schedule, then deliberately retained the historical prefix. Original sequence IDs
were kept for FASTA byte comparisons; actual replay labels are recorded separately.
No failed upload, incomplete client or unsuccessful invocation was counted as a pass.

See [machine-readable comparisons](verification/original_library_replay_20260929/original_library_replay_summary.json),
[raw curation comparisons](verification/original_library_replay_20260929/raw_rebuild_comparison.json),
[forest refit](verification/original_library_replay_20260929/raw_evaluator_refit_comparison.json)
and [replay instructions](RESEARCH_REPLAY.md). Computational reproduction does not settle
DRAMP/dbAMP source-use questions or establish experimental peptide activity.
