# Shared evaluator model cards

Generated from the checked-in data by `shared-evaluator/scripts/train_evaluator_models.py`.
See [`training_data_summary.json`](training_data_summary.json) for input hashes and counts.
These retrospective models are research screening tools, not validated clinical or
prospective predictors.

## AMP likelihood classifier

- Model: 50-tree Random Forest; 27 sequence and biophysical features.
- Positive class: curated AMP sequences in the evaluator view.
- Negative class: UniProt 25%-identity sequences, which are unlabeled proteins rather
  than experimentally confirmed non-AMPs.
- Rows (balanced): train 53398; validation 810;
  test 810.
- Validation: ROC-AUC 0.9733; average precision 0.9787; Brier score 0.0621.
- Test: ROC-AUC 0.9762; average precision 0.9811; Brier score 0.0580.

## Empirical human erythrocyte hemolysis classifier

- Model: 50-tree Random Forest with balanced class weights; 27 sequence and
  biophysical features.
- Target: HC50 < 100 µM from an explicit `50% Hemolysis` assay on human erythrocytes.
- Concentrations are converted from supported units to µM using peptide molecular
  weight. Censored or uncertainty intervals crossing 100 µM are unlabeled; sequences
  with conflicting labeled observations are excluded.
- Eligible assay rows: 285; labeled rows:
  239; unambiguous unique peptides:
  183 (105 positive,
  78 negative); similarity groups:
  94.
- Evaluation: 5-fold out-of-fold predictions; no >=80% Levenshtein component spans folds.
- Out-of-fold metrics: ROC-AUC 0.7280; average precision 0.7576; Brier score 0.2113.

The split groups sequences connected by normalized Levenshtein similarity >= 0.80,
which reduces close-sequence leakage. The limited and heterogeneous assay corpus
still does not establish prospective performance.

## Biophysical membrane-disruption proxy

`biophysical_membrane_disruption_proxy` is a separate analytical heuristic using
hydropathy and hydrophobic moment. It is not an empirical assay model and is not
included in the hemolysis model's training target.

## Uncertainty and limitations

The runtime reports standard deviation across trees as ensemble spread. It is not a
calibrated confidence interval or a guarantee of out-of-domain detection. The exact
input hashes, class counts, split sizes, and known limitations are recorded in the
training summary.
