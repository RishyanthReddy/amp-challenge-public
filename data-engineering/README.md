# AMP Challenge 2027 — Data Engineering

Reproducible, provenance-preserving data preparation for antimicrobial peptide (AMP) generation and downstream modelling. This project deliberately stops at the model-ready data layer: it does not train or generate peptides.

## What it produces

The pipeline reads immutable local source data from `data/raw/` and writes derived artifacts only:

- Canonical sequences with conservative normalization and challenge-validity flags.
- Record-level provenance retaining every source file, ID, header, original sequence, and normalized sequence.
- Separate activity, toxicity, and annotation tables; assay observations are not collapsed into one label.
- Challenge-reference overlap flags, similarity clusters, QC reports, figures, and deterministic splits.
- Parquet model-ready tables with UTF-8 CSV counterparts for inspection.
- Seven reproducible biophysical descriptors, including charge, pI, molecular weight, hydrophobic moment, Boman index, GRAVY, and instability index.
- A MIC-supported motif library, curated 100-seed panel, and generator-side synthesis feasibility filter.

The challenge reference is used only for novelty validation and exclusion from the generation corpus; it is not a declared training source.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

Install the locked environment:

```powershell
uv sync
```

## Rebuild and verify

```powershell
uv run python -m amp_data.build
uv run pytest
uv run python -m amp_data.validate data\processed\unique_amp_sequences.fasta
uv run python scripts\write_release_checksums.py
```

Use `--top100` when validating a ranked top-100 FASTA; it additionally rejects a sequence whose similarity to any challenge-reference peptide is strictly greater than 80%.

```powershell
uv run python -m amp_data.validate generate\top.fasta --top100
```

The build is deterministic and never modifies `data/raw/`.

## Generator handoff artifacts

- `data/processed/views/` contains explicit autoregressive, VAE/latent, diffusion, evolution-seed, and evaluator views.
- `data/processed/curated_seed_panel.csv` contains 100 activity-supported, biophysically filtered, diversity-selected seed sequences.
- `data/processed/motif_library.csv` contains recurring functional motifs mined from full-training sequences with local MIC observations.
- `data/reports/biophysical_descriptor_report.csv` summarizes descriptor distributions for the core training corpus.
- `src/amp_data/synthesis_filter.py` exposes `is_synthesizable(sequence)` and `filter_sequences(sequences)` for early generator-loop filtering.

```python
from amp_data.synthesis_filter import filter_sequences, is_synthesizable

is_valid, reason = is_synthesizable("AKRKLVWQ")
batch = filter_sequences(["AKRKLVWQ", "AAAAAAXX"])
```

`warn_unpaired_cys` is a soft warning; other non-`ok` outcomes are rejection reasons.

## Current build summary

| Metric | Result |
| --- | ---: |
| Source-standardized records | 122,581 |
| Unique canonical sequences | 38,681 |
| Core L/unspecified AMP training pool | 27,509 |
| Challenge-safe generation corpus | 4,051 |
| Novelty-safe train / validation / test | 3,241 / 405 / 405 |
| MIC-supported seed candidates meeting biophysical filters | 831 |
| Curated diverse seed panel | 100 |
| Functional motifs meeting recurrence rule | 4,950 |

The eligible corpus is compared exhaustively with the same Indel-ratio semantics used by the official validator. Its 6,854,993 feasible pair comparisons produce clusters at 80%, 70%, and 60% similarity. The 80% components remain intact during splitting to prevent leakage. The primary `train.parquet` excludes validation and test sequence IDs; reference-filtered split files are explicitly named `novelty_safe_*`.

## Layout

```text
configs/data_config.yaml        Fixed paths, constraints, and seed
src/amp_data/                   Parsers, normalization, validation, build CLI
data/raw/                        Immutable source exports
data/challenge/                 Official novelty-reference FASTA
data/processed/                 Model-ready Parquet/CSV tables and FASTA corpus
data/processed/views/           Role-specific generator/evaluator views
data/processed/curated_seed_panel.csv  Diverse MIC-supported generator seeds
data/processed/motif_library.csv       Functional motif conditioning library
data/reports/                   Inventories, QC, overlap, split, and cluster reports
data/DATA_CARD.md               Dataset scope and limitations
data/PROVENANCE.md              End-to-end lineage policy
tests/                          Automated tests
```

## Data policy

- Raw files are never edited or normalized in place.
- Normalization removes formatting whitespace and uppercases text; it never repairs residues or modifications.
- Non-standard residues, invalid lengths, metadata-indicated modifications, and lowercase (suspected D/mixed-stereochemistry) records are flagged.
- Identical sequences map to one canonical ID while retaining all provenance records.
- MIC and toxicity values retain their raw values and units; incompatible assays are not silently converted or merged.
- APD3 is not a standalone local source. Do not claim direct APD3 use without a licensed, versioned export.
- The curated seed panel uses all validated training sequences with local MIC evidence; it does not claim natural origin where local provenance cannot establish it.
- Biophysical descriptors use hard-coded reference scales and are reproducible without external descriptor libraries.

See [the data card](data/DATA_CARD.md), [provenance documentation](data/PROVENANCE.md), and [the QC report](data/reports/data_quality_report.md) for details.

## Official challenge alignment

The implementation follows the current [AMP Challenge 2027 repository](https://github.com/szczurek-lab/amp-challenge-2027): standard amino-acid alphabet, length 8–50, uniqueness, no exact library overlap with the antibacterial reference, and the stricter top-100 similarity condition.

## Next step

Use `data/processed/generation/amp_sequences.fasta` as the clean corpus, `curated_seed_panel.csv` for seed-based generation, `motif_library.csv` for motif conditioning, and the separate activity/toxicity tables for downstream objectives. Keep modelling, ranking, and submission code outside this data-engineering layer.
