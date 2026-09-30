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
five or more, and net charge below +1. Biophysical features include free-terminal charge
at pH 7.4, Boman index and GRAVY.
The Eisenberg hydrophobic moment uses a 100-degree residue angle and the maximum over
11-residue windows (or the whole sequence when shorter). The legacy field name
`net_charge_ph7` refers to the pH-7.4 calculation.

The shared evaluator uses 27 sequence/biophysical features for AMP and empirical RBC
hemolysis prediction. The hemolysis forest is trained on 183 unambiguous peptides with
human-erythrocyte assay labels. APEX scores eligible ranked candidates across 11 pathogens.
Auxiliary library candidates receive AMP/safety predictions but no APEX MIC predictions.

Top candidates must have maximum Levenshtein ratio <=0.80 against the challenge's
39,448-sequence antibacterial reference. This is the validator's ratio definition, not a
claim that every possible sequence-alignment identity metric is equivalent.

## Quality and diversity

MAP-Elites discretizes charge, hydrophobic moment and length into 150 cells and assigns a
quality score to every eligible candidate. The DPP receives the full annotated pool, rather
than only the best sequence in each cell. Quality rewards AMP likelihood and low predicted
MIC, while penalizing toxicity and tree-to-tree prediction spread; this is a heuristic,
not a calibrated statistical confidence bound.

The DPP similarity kernel combines an RBF kernel over standardized biophysical features
(weight 0.60) with cosine similarity of normalized amino-acid composition (weight 0.40).
Selection also enforces reference novelty, internal Levenshtein ratio at or below 0.80,
composition cosine similarity at or below 0.92, and representation of both generator
families with a maximum of 60 candidates per family.
The selected Top 50 is the first 50 entries of the ranked Top 100.

The submitted library combines 3,964 eligible scored candidates with 46,036 ProGen2
auxiliary sequences screened for perplexity, uniqueness, reference exclusion and synthesis.
All Top 100 sequences are included. The ranked list contains 60 autoregressive and 40
evolutionary candidates.

## Reproducibility and interpretation

The root `uv run generate` runs inference, mutation search, scoring, selection and assembly
from released weights with fixed schedules. It requires the declared RTX 4090/runtime and
checks the submitted FASTA hashes before replacing outputs. `uv run export_submission` is
the separate CPU export of saved tables. Training code is also provided. See
[model generation](../generation/README.md). Full replay reproduced the checkpoint, evaluator
models, evolutionary outputs and the original ranked lists; an additional fresh 50,000
library build passed all sequence checks. See [verification](FULL_REPLAY_VERIFICATION.md).

Activity, safety and MIC are model predictions. No generated sequence has been experimentally
validated by this project. The source data, training splits and label limitations are
recorded in the [data card](../data-engineering/data/DATA_CARD.md) and
[model cards](../shared-evaluator/reports/model_cards.md).
