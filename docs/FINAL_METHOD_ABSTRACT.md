# Method abstract

We developed an antimicrobial-peptide design pipeline combining a fine-tuned ProGen2
language model with evolutionary search. New sequences are screened for amino-acid validity,
length, novelty and synthesis feasibility. A shared evaluator uses 27 sequence and
physicochemical features to predict AMP likelihood and human-erythrocyte hemolysis risk.
The APEX ensemble predicts MIC across 11 pathogens for candidates eligible for ranking.
MAP-Elites assigns candidates to charge, hydrophobic-moment and length niches and computes
a quality score that penalizes toxicity and ensemble spread. A constrained determinantal
point process selects a diverse ranked Top 100 from the annotated candidate pool, with a
nested Top 50.

The library contains 50,000 unique peptides: 3,964 scored autoregressive and evolutionary
candidates and 46,036 auxiliary ProGen2 sequences. Auxiliary sequences passed a perplexity
screen at or below 100, duplicate and exact-reference checks, and the synthesis filter.
The Top 100 contains 60 autoregressive and 40 evolutionary candidates, all drawn from the
library. There are no exact library matches to the 39,448 reference sequences. Maximum
Top-100/reference and internal Top-100 Levenshtein ratios are 0.80 and 0.7778, respectively.
Auxiliary peptides do not have APEX MIC scores.

The empirical hemolysis classifier was trained on 183 peptides with unambiguous human
erythrocyte HC50 labels. Grouped five-fold evaluation yielded out-of-fold ROC-AUC 0.7280.
These retrospective results do not establish prospective performance. Activity, hemolysis
and MIC values are predictions; none of the newly generated peptides has been assayed by
this project.

HydrAMP and AMP-Diffusion were explored as research baselines. Their generated sequences
are excluded from this integrated ProGen2/evolution entry. The public repository provides
the [fine-tuned checkpoint](https://github.com/RishyanthReddy/amp-challenge-public/releases/tag/progen2-checkpoint-20260928),
inference code and deterministic export of the selected sequences. Training sources,
computational filters and historical data-access gaps are described in the
[data disclosure](PUBLIC_TRAINING_DISCLOSURE.md) and [method](method.md).
