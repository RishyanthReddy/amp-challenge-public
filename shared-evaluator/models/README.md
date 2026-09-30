# Original AMP and hemolysis forests

These are the two models used to score the submitted library. Their bytes reproduced
exactly when the evaluator was retrained from the archived inputs on 2026-09-28.
The obsolete synthetic toxicity regressor is not part of this model pair.

Verify the fingerprints without loading either file:

```bash
uv run python scripts/verify_evaluator_assets.py
```

Load and score a fixed peptide in the original locked environment:

```bash
uv run --project shared-evaluator --locked --python 3.12 python \
  scripts/verify_evaluator_assets.py --smoke
```

Both forests use 50 trees and 27 sequence/biophysical features. The model manifest
records their hashes and original runtime. The AMP model compares curated AMPs to
unlabeled UniProt negatives. The human-RBC model predicts HC50 < 100 µM and has
grouped out-of-fold ROC-AUC 0.7280 on 183 peptides. These are retrospective research
models; their predictions are not measured activity or safety for the submitted peptides.

Read the [model cards](../reports/model_cards.md),
[training summary](../reports/training_data_summary.json) and
[source disclosure](../../docs/PUBLIC_TRAINING_DISCLOSURE.md).
Original project software uses MIT; third-party inputs retain their source terms.
Including model files does not resolve the disclosed DRAMP/dbAMP source-use questions.
