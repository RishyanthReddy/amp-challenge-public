# Model assets and attribution

## ProGen2 checkpoint

The fine-tuned checkpoint is distributed through the
[public release](https://github.com/RishyanthReddy/amp-challenge-public/releases/tag/progen2-checkpoint-20260928).
Download and verify it with:

```bash
uv run python cloud/fetch_progen_checkpoint.py
```

The fetcher checks configuration, generation configuration, tokenizer and weights against
recorded sizes and SHA-256 hashes. All four release assets have also been downloaded
anonymously and checked. GitHub CLI may request login; browser downloads are public.
Place manually downloaded files in
`autoregressive-models/checkpoints/progen2_small_amp_best_val/` and run the fetcher to
verify the cache.

Model SHA-256:
`124b8ea7df5c96cda927bada51c9d26d89f91636d0975fe70e7bc0ef182f9f83`.
The pinned base model/code revision is `43237a0b733c6629226a079266d2985c9fdce9b7`.
The [ProGen2-small mirror](https://huggingface.co/hugohrban/progen2-small) declares
BSD-3-Clause; its notice is retained in [licenses/PROGEN_BSD_3_CLAUSE.txt](../licenses/PROGEN_BSD_3_CLAUSE.txt).

## Evaluators and research baselines

| Component | Availability | Purpose |
| --- | --- | --- |
| AMP / RBC forests | Not bundled | Required to repeat activity and hemolysis scoring |
| APEX | Eight weights and source in `diffusion-models/apex/` | Pathogen MIC prediction; weights match upstream fingerprints and retain the MIT notice |
| HydrAMP | Source adapter only; checkpoint and decomposer omitted | Research baseline; excluded from submitted sequence ancestry |
| AMP-Diffusion | Source and hash-pinned checkpoint download helper | Research baseline; excluded from submitted sequence ancestry |

APEX uses a separate locked environment. Its asset provenance is recorded in
[APEX_WEIGHT_PROVENANCE.json](license_evidence/APEX_WEIGHT_PROVENANCE.json).
Read the [HydrAMP](../vae-latent-models/README.md) and
[diffusion](../diffusion-models/README.md) guides for baseline requirements.

## Data and licenses

Training sources include DBAASP, DRAMP, dbAMP and AMPlify. UniProt-derived negative
partitions support the AMP evaluator. The exact AMPlify training FASTA matches the authors'
CC BY 4.0 Zenodo deposit and is included with [attribution](../data-engineering/data/raw/amplify/ATTRIBUTION.md).
This dataset license is separate from AMPlify's GPL software license.

DBAASP attribution: this project uses peptide and assay data from the Database of
Antimicrobial Activity and Structure of Peptides. Cite Pirtskhalava et al., *DBAASP v3*,
Nucleic Acids Research (2021), DOI: [10.1093/nar/gkaa991](https://doi.org/10.1093/nar/gkaa991).

The root MIT license covers original project code. Third-party data, weights and software
retain their own terms. See [third-party notices](../THIRD_PARTY_NOTICES.md),
[source access](DATA_ACCESS.md), [training disclosure](PUBLIC_TRAINING_DISCLOSURE.md) and
[evaluator model cards](../shared-evaluator/reports/model_cards.md).
