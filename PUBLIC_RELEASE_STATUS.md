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
| Original AMP/RBC forests | `shared-evaluator/models/` | Scoring with the two original, hash-pinned research models |
| Retained DBAASP exports | `data-engineering/data/raw/DBAASP/` | Exact peptide and assay inputs, with acknowledgment |
| Retained DRAMP General snapshot | `data-engineering/data/raw/DRAMP/` | Exact historical General inputs under source terms and attribution |
| Pinned public source downloads | [Data access](docs/DATA_ACCESS.md) | Exact dbAMP, DRAMP antibacterial and UniProt inputs |
| Exact AMPlify training FASTA | `data-engineering/data/raw/amplify/` | The verified CC BY 4.0 snapshot, with attribution |
| Verification records | [Documentation](docs/README.md) | Export, inference and historical replay results with their scopes |

## Additional inputs for retraining

DRAMP antibacterial/patent and dbAMP raw snapshots, mixed-source curated training tables and baseline
checkpoints are not bundled. The original AMP/RBC forests are included. Raw inputs can be restored and curated views rebuilt using the documented commands. Exact historical selection also uses the archived audited/APEX-scored candidate pool. The [data access guide](docs/DATA_ACCESS.md)
records expected files and hashes. All 11 original raw files are retained in the private archive. Exact public retrieval is now
pinned where verified; the two retained DRAMP General files are included with attribution. Original queries,
acquisition dates and some release versions remain unknown.

The root `uv run generate` exports the frozen selected tables; it does not launch training
or sample a new library. A historical training-to-selection replay reproduced the checkpoint
and ranked lists from archived inputs. A separate fresh full-size library also passed the
sequence checks. The 2026-09-29 follow-up reproduced the original auxiliary pools, library Parquet and both FASTAs exactly. See [the follow-up](docs/ORIGINAL_LIBRARY_REPLAY.md).
See the [replay report](docs/FULL_REPLAY_VERIFICATION.md).

## Scientific and submission scope

The entry contains ProGen2 and evolutionary sequences. HydrAMP and AMP-Diffusion code is
retained for comparison, and their baseline outputs are excluded from the submitted library.
All activity, safety and MIC values are predictions; this project has not assayed its
newly generated peptides.

Public code and checkpoint availability do not settle the remaining source-data questions
or establish co-authorship eligibility. The source disclosure records those limitations;
the organizers make the eligibility decision.
