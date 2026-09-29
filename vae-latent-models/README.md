# HydrAMP research baseline

HydrAMP is a conditional variational autoencoder for antimicrobial-peptide design.
This directory retains the starter-kit generator and the project's model adapter for
baseline experiments. HydrAMP-generated sequences are excluded from the submitted
ProGen2/evolution library and Top 100.

## What is included

| Path | Purpose |
| --- | --- |
| `src/vae_latent/model_adapter.py` | Latent sampling, AMP/MIC conditioning and decoder adapter |
| `src/hydramp_starter_kit/generate.py` | Standalone baseline library generation with bounded collection rounds |
| `scripts/validate_prototypes.py` | Prototype analysis |
| `docs/` and `outputs/` | Recorded baseline settings and asset fingerprints |
| `pyproject.toml` and `uv.lock` | The legacy Python 3.8 / TensorFlow 2.2 runtime |

Checkpoint files, the PCA decomposer, copied training records and experimental tables are
not bundled in this public checkout. The documentation records historical experiments;
it does not mean those data files are present here.

## Repeating a baseline experiment

Obtain the model and decomposer from the
[organizer starter kit](https://github.com/szczurek-lab/hydramp-starter-kit), and check them
against the [recorded asset fingerprints](docs/hydramp_setup.md). The original inference
source is [szczurek-lab/hydramp](https://github.com/szczurek-lab/hydramp), pinned in this
component's dependency configuration.

The standalone command expects these paths relative to `vae-latent-models/`:

- `checkpoint/model/` — HydrAMP model.
- `checkpoint/pca_decomposer.joblib` — PCA decomposer.
- `data/antibacterial.fasta` — challenge reference; use the repository's file at
  `data-engineering/data/challenge/antibacterial.fasta` or pass that path explicitly.

With those assets and a compatible legacy environment available:

```bash
cd vae-latent-models
uv run generate_broad_spectrum \
  --antibacterial-fasta ../data-engineering/data/challenge/antibacterial.fasta \
  --n-sequences 100 --top-k 10 --seed 42
```

This is a baseline example, not a fresh-clone quick start. The legacy dependency stack
requires a compatible platform and has not been revalidated as part of the public release.
Run baseline experiments in a separate checkout. The command writes to this component's
`generate_broad_spectrum/` directory, separate from the root submission files.

## Method and interpretation

HydrAMP conditions latent decoding on antimicrobial activity and low MIC. The standalone
starter-kit route uses classifier filtering, sequence constraints, exact-reference
exclusion and an additional synthesizability/novelty screen for its ranked list. Its
filters differ from the shared synthesis filter used by our final entry.

Historical settings and results are recorded in [model scope](docs/model_scope.md),
[prototype selection](docs/prototype_selection.md) and
[baseline candidate analysis](docs/vae_candidates_audit.md). These are baseline records,
not measurements of the submitted peptides or a second competition entry.

## Attribution

HydrAMP: Szymczak et al., *Nature Communications* 14, 1453 (2023),
[doi:10.1038/s41467-023-36994-z](https://doi.org/10.1038/s41467-023-36994-z).
The retained component license is [MIT](LICENSE). External data and model assets retain
their original terms; see [source access](../docs/DATA_ACCESS.md).
