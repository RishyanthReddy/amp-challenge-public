# Method

## Candidate generation

The selected entry combines ProGen2-small fine-tuning with a family-aware evolutionary
search. ProGen2 samples amino-acid tokens autoregressively. Evolution applies bounded
mutations to seed peptides and records the parent-child graph. Both methods produce linear
sequences using the 20 standard amino acids, with a target length range of 8–50 residues.

HydrAMP and AMP-Diffusion are retained as comparative research implementations. Sequences
with ancestry from their official baseline checkpoints are excluded from the submitted entry.

## Screening and prediction

Sequence checks remove invalid candidates and exact antibacterial-reference matches.
Synthesizability rules reject homopolymers of four or more residues, hydrophobic runs of
five or more, and net charge below +1. Biophysical features include pH-7 charge,
hydrophobic moment, Boman index and GRAVY.

The shared evaluator uses 27 sequence/biophysical features for AMP and empirical RBC
hemolysis prediction. The hemolysis forest is trained on 183 unambiguous peptides with
human-erythrocyte assay labels. APEX scores eligible ranked candidates across 11 pathogens.
Auxiliary library candidates receive AMP/safety predictions but no APEX MIC predictions.

Top candidates must have maximum Levenshtein ratio <=0.80 against the challenge's
39,448-sequence antibacterial reference. This is the validator's ratio definition, not a
claim that every possible sequence-alignment identity metric is equivalent.

## Quality and diversity

MAP-Elites discretizes charge, hydrophobic moment and length into 150 cells. Its quality
score penalizes predictor uncertainty. A constrained DPP optimizer then balances quality,
feature-space diversity, model-family coverage and explicit sequence redundancy limits.
The selected Top 50 is the first 50 entries of the ranked Top 100.

The submitted library combines 3,964 eligible scored candidates with 46,036 ProGen2
auxiliary sequences screened for perplexity, uniqueness, reference exclusion and synthesis.
All Top 100 sequences are included. The ranked list contains 60 autoregressive and 40
evolutionary candidates.

## Reproducibility and interpretation

The root entry point exports frozen selected tables deterministically. Separate model
inference and training code is provided. Full replay reproduced the checkpoint, evaluator
models, evolutionary outputs and the original ranked lists; an additional fresh 50,000
library build passed all sequence checks. See [verification](FULL_REPLAY_VERIFICATION.md).

Activity, safety and MIC are model predictions. No generated sequence has been experimentally
validated by this project. The source data, training splits and label limitations are
recorded in the [data card](../data-engineering/data/DATA_CARD.md) and
[model cards](../shared-evaluator/reports/model_cards.md).
