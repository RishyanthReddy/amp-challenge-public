# Final private-entry method abstract

We developed a computational pipeline for antimicrobial-peptide design. For the
current conservative ranked portfolio, candidate generation uses a participant fine-tuned
ProGen2 autoregressive model and project evolutionary search; official HydrAMP and
AMP-Diffusion checkpoint outputs remain in the research workspace but are excluded from the
ranked portfolio and library. A shared evaluator extracts 27 sequence and physicochemical
features, predicts AMP likelihood and human-erythrocyte hemolysis risk, and applies the
official APEX predictor across 11 pathogen models to eligible candidates. MAP-Elites
quality-diversity screening over charge, hydrophobic moment, and length feeds a constrained
positive-semidefinite-kernel determinantal point process to select nested Top 50 and Top 100
lists.

The local library contains 50,000 unique peptides: 3,964 scored autoregressive/evolutionary
candidates and 46,036 auxiliary ProGen2 sequences. Auxiliary sequences passed a ProGen2
perplexity screen at `<= 100`, duplicate and exact-reference checks, and the project
synthesis filter. The Top 100 contains 60 autoregressive and 40 evolutionary candidates.
It is fully contained in the library. The library has zero exact matches to the 39,448
reference sequences; maximum Top-100/reference and internal Top-100 Levenshtein ratios are
`0.80` and `0.7778`, respectively. Activity, hemolysis, and MIC fields are model predictions,
not assay results; auxiliary peptides do not have APEX MIC scores.

The empirical hemolysis model was trained from 183 unambiguous peptides with explicit human
erythrocyte assay labels. Its grouped five-fold out-of-fold ROC-AUC is 0.7280. This small,
retrospective evaluation does not establish prospective performance. No peptide generated
by this project has been experimentally assayed by the team.

The submission is declared as one integrated ProGen2/evolution method. Training sources and filters are disclosed in `data-engineering/data/DATA_CARD.md` and `docs/training_source_scope.json`. Existing model weights are available in the private GitHub prerelease. Root generation reproduces frozen artifacts; separate inference code and verification are provided. Data use and evaluator access are owner-confirmed, not independently legally certified.
