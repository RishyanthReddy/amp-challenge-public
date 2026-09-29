# Model assets and attribution

> License review update (2026-09-29): the exact AMPlify training FASTA matches the authors' CC BY 4.0 Zenodo deposit 7320306. Older statements below describing its data license as unestablished are superseded by this finding. See the public training disclosure and AMPLIFY_DATA_LICENSE_MATCH.json. Other source questions remain open.

## ProGen2 checkpoint

The fine-tuned checkpoint is distributed through the public GitHub release
[`progen2-checkpoint-20260928`](https://github.com/RishyanthReddy/amp-challenge-public/releases/tag/progen2-checkpoint-20260928).
Download it with:

```bash
uv run python cloud/fetch_progen_checkpoint.py
```

The fetcher checks all four assets: configuration, generation configuration, tokenizer and
weights. Model SHA-256:
`124b8ea7df5c96cda927bada51c9d26d89f91636d0975fe70e7bc0ef182f9f83`.
Weights are cached under `autoregressive-models/checkpoints/`, outside Git tracking.
The pinned base model/code revision is `43237a0b733c6629226a079266d2985c9fdce9b7`.

Base model: [ProGen2-small mirror](https://huggingface.co/hugohrban/progen2-small),
with a declared BSD-3-Clause license; see its linked upstream sources and notices.

## Evaluators and research baselines

| Component | Location | Purpose |
| --- | --- | --- |
| AMP / RBC forests | Not bundled | Required to rerun activity/hemolysis scoring |
| APEX | `diffusion-models/apex/` | Eight weights included and checked against upstream hashes; MIT notice retained |
| HydrAMP | `vae-latent-models/` | Research baseline; excluded from submitted sequence ancestry |
| AMP-Diffusion | `diffusion-models/` | Research baseline; excluded from submitted sequence ancestry |

The APEX environment is isolated from the root and AR environments. Its source and weight
hashes are recorded in the evaluation manifests. Baseline asset instructions are in their
component READMEs.

## Data sources

The curated corpus records DBAASP, DRAMP, dbAMP and AMPlify provenance. UniProt-derived
negative partitions are used by the AMP evaluator. See the
[data card](../data-engineering/data/DATA_CARD.md),
[source manifest](TRAINING_SOURCE_MANIFEST.csv), and
[evaluator training summary](../shared-evaluator/reports/training_data_summary.json).

DBAASP attribution: this project uses peptide and assay data from the Database of
Antimicrobial Activity and Structure of Peptides. Cite Pirtskhalava et al., *DBAASP v3*,
Nucleic Acids Research (2021), DOI: [10.1093/nar/gkaa991](https://doi.org/10.1093/nar/gkaa991).
Retain each source's attribution and applicable terms when reusing its data.

The root MIT license covers original project code. It does not relicense third-party data,
weights or software. Source access and historical gaps are disclosed in [DATA_ACCESS.md](DATA_ACCESS.md).

Anonymous downloads are available directly from the [public release](https://github.com/RishyanthReddy/amp-challenge-public/releases/tag/progen2-checkpoint-20260928). Place the four model/config/tokenizer files under `autoregressive-models/checkpoints/progen2_small_amp_best_val/`; the fetch helper verifies cached files without network access.
