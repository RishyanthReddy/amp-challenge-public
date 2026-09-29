# Training provenance metadata

These files disclose record membership without copying peptide sequences or raw assay measurements. Sequence hashes use SHA-256 of the normalized uppercase sequence. Multiple source links are not independent training examples.

| File | Contents |
| --- | --- |
| AR_TRAINING_SOURCE_IDS.csv | 26,699 AR training sequences and 94,314 source links |
| PROVENANCE_SUMMARY.json | AR input fingerprints and coverage checks |
| EVALUATOR_AMP_MEMBERSHIP.csv | Ordered AMP classifier examples, labels and train/validation/test partitions |
| EVALUATOR_AMP_SOURCE_IDS.csv | Source IDs for positive examples and UniProt-derived negatives |
| EVALUATOR_RBC_MEMBERSHIP.csv | 183 RBC peptides, labels, similarity groups and held-out folds |
| EVALUATOR_RBC_ASSAY_IDS.csv | 209 retained supporting assay rows and provider IDs |
| EVALUATOR_DISCLOSURE_SUMMARY.json | Input hashes, runtime versions, counts and checks |

AMP membership contains 53,398 training, 810 validation and 810 test examples. These partitions are sequence-disjoint. Negative examples are UniProt-derived HydrAMP records, not experimentally confirmed inactive peptides. `source_row` and `partition_row` are zero-based; source IDs preserve the original file's identifiers.

RBC membership reproduces the training script's unambiguous HC50 labels. The 183 peptides are supported by 209 retained assay rows. The script labels 239 rows before excluding sequences with conflicting labels; these counts describe different filtering stages. All 183 peptides train the final forest. `held_out_fold` describes retrospective cross-validation, not a separate final-model test set.

## Reconstruct the evaluator disclosure

Supply a checkout with the exact archived training inputs listed in the summary:

```bash
uv run --project shared-evaluator --locked python scripts/disclose_evaluator_membership.py \
  --source-root /path/to/archived-project \
  --output-dir /path/to/disclosure-output
```

This command loads the training script's membership helpers without fitting models. It checks the recorded input fingerprints and fold sizes. It does not recompute performance metrics or establish the availability of historical provider exports. The source checkout must be trusted because its Python module is imported.

See [source access](../DATA_ACCESS.md). The exact AMPlify snapshot is included with CC BY 4.0 attribution. Other source snapshots remain provider-hosted; missing acquisition dates, queries and versions are still disclosed limitations.
