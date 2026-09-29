# Data lineage

`data/raw/` (immutable) → source-specific FASTA/CSV/XLSX parsing → normalized record-level provenance → canonical exact sequence table → independent activity, toxicity, and annotations tables → exact-identity clusters → deterministic train/validation/test views.

The pipeline writes model-ready tables as Parquet (Zstandard-compressed) and companion UTF-8 CSV files for transparent inspection. Re-running `uv run python -m amp_data.build` regenerates only derived outputs and never writes into `data/raw/`.

Challenge-reference policy: `data/challenge/antibacterial.fasta` is used only to flag/exclude exact matches from `unique_amp_sequences.fasta` and to validate generated libraries. It is not concatenated into the training corpus.
