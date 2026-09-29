# What this release contains

This is the public repository for the ProGen2 and evolutionary-search entry. It provides
the selected sequences, deterministic export, ProGen2 inference, source code and data
provenance. The private repository preserves the wider research archive.

## Available here

| Material | Location | What it supports |
| --- | --- | --- |
| Submitted library and ranked candidates | `generate/` | 50,000 unique sequences and their ranked Top 100 |
| Frozen selected tables | `portfolio-selection/outputs/` | Repeatable export of the submitted FASTAs |
| Fine-tuned ProGen2 checkpoint | [Public release](https://github.com/RishyanthReddy/amp-challenge-public/releases/tag/progen2-checkpoint-20260928) | Sampling new candidates; all four assets were downloaded anonymously and hash-checked |
| Generation and selection code | Component directories | The research method and its individual stages |
| APEX predictor | `diffusion-models/apex/` | Eight model files matching the upstream starter-kit fingerprints |
| Training-source metadata | [Source disclosure](docs/PUBLIC_TRAINING_DISCLOSURE.md) | Source identifiers, sequence hashes, partitions and known acquisition gaps |
| Exact AMPlify training FASTA | `data-engineering/data/raw/amplify/` | The verified CC BY 4.0 snapshot, with attribution |
| Verification records | [Documentation](docs/README.md) | Export, inference and historical replay results with their scopes |

## Additional inputs for retraining

Most raw database snapshots, curated training tables, AMP/RBC forest files and baseline
checkpoints are not bundled. Full training and evaluator scoring therefore require inputs
from the research archive or their original providers. The [data access guide](docs/DATA_ACCESS.md)
records expected files and hashes. Some historical queries, acquisition dates and versions
are unknown, so current provider downloads cannot be assumed to reconstruct the same inputs.

The root `uv run generate` exports the frozen selected tables; it does not launch training
or sample a new library. A historical training-to-selection replay reproduced the checkpoint
and ranked lists from archived inputs. A separate fresh full-size library also passed the
sequence checks. Exact resampling of every original auxiliary library sequence was not tested.
See the [replay report](docs/FULL_REPLAY_VERIFICATION.md).

## Scientific and submission scope

The entry contains ProGen2 and evolutionary sequences. HydrAMP and AMP-Diffusion code is
retained for comparison, and their baseline outputs are excluded from the submitted library.
All activity, safety and MIC values are predictions; this project has not assayed its
newly generated peptides.

Public code and checkpoint availability do not settle the remaining source-data questions
or establish co-authorship eligibility. The source disclosure records those limitations;
the organizers make the eligibility decision.
