# Final data-engineering audit

Audit date: 2026-08-24. This report was generated after rebuilding the pipeline and querying the produced Parquet files. Raw inputs were not modified.

## Result

All requested checks passed. The only intentional limitation is that the classification view remains explicitly positive-only; no HydrAMP or external negative-control dataset was added.

## 1. Validated AMP corpus: 27,662 → 27,509

`27,662` is the counterfactual count obtained from the current canonical table when applying alphabet, length, and known-modification rules **before** the new lowercase/stereochemistry exclusion. The new core training pool is `27,509`.

The difference is exactly **153** sequences: they pass the other three rules but contain lowercase residue notation and are now labelled `suspected_d_or_mixed` rather than silently converted to ordinary L-peptides.

### Canonicalization and filters

| Stage / reason | Count |
| --- | ---: |
| Source-standardized records | 122,581 |
| Unique canonical normalized sequences | 38,681 |
| Record-to-canonical surplus | 83,900 |
| Pre-stereochemistry validated sequences | 27,662 |
| Core validated training sequences | 27,509 |

The 83,900 surplus is **not** a claim that all are duplicate raw rows: it includes repeated assay observations and multiple source/provenance records mapped to the same canonical sequence.

The following non-exclusive raw flags overlap. They are shown to make the filtering transparent: invalid alphabet 2,063; invalid length 9,129; known modification 540; lowercase residue notation 327; lowercase records that otherwise passed the pre-stereochemistry rules 153.

For a mutually exclusive audit partition, rules are applied in this precedence order: invalid alphabet, invalid length, known modification, lowercase stereochemistry.

| Mutually exclusive outcome | Count |
| --- | ---: |
| Retained in core training | 27,509 |
| Invalid alphabet | 2,063 |
| Invalid length | 8,863 |
| Known modification | 93 |
| Lowercase / suspected D-or-mixed stereochemistry | 153 |
| Total | 38,681 |

## 2. Novelty-safe corpus: 4,126 → 4,051

The previous 4,126 count was the reference-safe corpus before applying the lowercase stereochemistry exclusion. The new count is 4,051.

| Exclusion from the previous novelty-safe corpus | Count |
| --- | ---: |
| Lowercase records that were otherwise valid and had no exact reference overlap | 75 |
| Remaining novelty-safe sequences | 4,051 |

No challenge-reference policy changed. The 75-sequence reduction is solely the new lowercase/D-or-mixed handling.

## 3. Core training view

`data/processed/train.parquet` contains exactly **27,509** rows. Every row has `valid_for_challenge = true`; it intentionally does **not** filter on `challenge_exact_overlap`.

- Rows with exact challenge-reference overlap retained in this training pool: **23,458**.
- This is correct: the official reference constrains generated/submitted candidates, not the known-AMP learning corpus.

Its schema is: `sequence_id`, `sequence`, `sequence_normalized`, `length`, `alphabet_valid`, `length_valid`, `invalid_residues`, `has_lowercase_residues`, `stereochemistry_status`, `modification_status`, `modification_reason`, `valid_for_challenge`, `challenge_exact_overlap`, `eligible_for_generation_corpus`.

## 4. Novelty-safe split and leakage audit

The separate novelty-safe views are derived from the 4,051 challenge-safe sequences:

| File | Sequences |
| --- | ---: |
| `novelty_safe_train.parquet` | 3,241 |
| `novelty_safe_validation.parquet` | 405 |
| `novelty_safe_test.parquet` | 405 |

All pairwise sequence-set intersections are zero. The 80% similarity graph has 1,552 connected components, constructed by exhaustive RapidFuzz Indel-ratio comparison. A component is assigned to exactly one split.

- Cross-split 80%-cluster leakage: **0 clusters**.
- Exact sequence leakage: **0 sequences**.

`sequence_clusters.parquet` schema: `sequence_id`, `cluster_id`, `cluster_size`, `threshold`, `cluster_method`.

## 5. Activity relationships

`activity.parquet` contains **20,491** source observations. `activity_prediction/*.parquet` is a genuine one-to-many assay view, not a copied sequence list:

- MIC observations attached to validated sequences: **12,329**.
- Activity-prediction rows across train/validation/test: **12,329**.
- Non-MIC rows in activity-prediction views: **0**.
- Missing joined sequence: **0**; missing target organism: **0**.
- Unique assayed sequences: **1,214**.
- Sequences with more than one observation: **1,076**.
- Largest observation count for one sequence: **242**.

Each activity-prediction row retains `sequence_id`, sequence, target organism/strain, source record IDs, assay row ID, raw assay text, assay type, measurement, units, and parsed MIC fields.

## 6. MIC parsing audit

`activity.parquet` retains `mic_value_raw` and `mic_unit_raw` for every MIC observation.

| Check | Count |
| --- | ---: |
| MIC observations | 14,104 |
| Raw MIC value retained | 14,104 |
| Exact numeric (`mic_qualifier = =`) | 9,938 |
| Censored (`<` or `>`) | 2,371 |
| Range | 1,795 |
| Unit/MW conversion available | 13,725 |
| Exact values with `mic_value_uM` | 9,593 |
| Rows with traceable molecular weight | 13,271 |
| Recomputed molecular-weight mismatches | 0 |

Ranges and censored values do not receive an invented exact `mic_value_uM`; they retain qualifier plus lower/upper bounds and, where unit conversion is valid, lower/upper µM bounds. Molecular weight is calculated deterministically from the joined normalized canonical sequence using the residue-mass table in `amp_data.core` plus free-terminal water mass.

## 7. Lowercase / D-amino-acid handling

All 327 canonical records with lowercase residue characters retain the original sequence in provenance, have `has_lowercase_residues = true`, and have `stereochemistry_status = suspected_d_or_mixed`. The 153 that would otherwise enter the validated pool are excluded. No flagged sequence is represented as an ordinary L-peptide in `train.parquet` or any novelty-safe view.

## 8–9. Reproducibility and tests

The complete build was run twice consecutively. Hashes matched between both outputs:

| Artifact | SHA-256 |
| --- | --- |
| `data/processed/sequence_clusters.parquet` | `0A741E69C67FE77796594AF8F1F680AA4184438F73DC02C81423EEE5EE33203F` |
| `data/processed/activity_prediction/train.parquet` | `69E548185382CA2ED5FDDCB614B784A5494D89B217F96F624C9B6DDDBB2CEC3E` |

The complete test suite was run with `uv run pytest`: **7 passed**.

## Commands

```powershell
uv run python -m amp_data.build
uv run pytest
uv run python -m amp_data.validate data\processed\unique_amp_sequences.fasta
```

## PDF-guide documentation completion

The version 1.0 handover additionally contains `data/schema.md`, `data/source_plan.md`, `data/example_rows.md`, `data/release_notes.md`, and `data/SHA256SUMS`; `reports/source_manifest.csv`, `normalization_report.csv`, `organism_mapping.csv`, `unit_conversion_rules.md`, `official_reference_matches.csv`, `dataset_splits.csv`, `leakage_check.csv`, and `assay_summary_policy.md`; and the five model-role views in `processed/views/`.

`reference_match_scope.md` explicitly documents the remaining version-1.0 boundary: exact official-reference matches are reported for the corpus, while nearest-reference similarity is computed only when validating generated top-100 candidates. `source_plan.md` documents that APD/APD3, HydrAMP, and negative controls are absent rather than silently substituted.
