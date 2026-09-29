# HydrAMP Starter Kit: AMP Challenge 2027 Baseline

A self-contained, reproducible generator that produces the **HydrAMP baseline library**
for the [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027).

HydrAMP ([Szymczak et al., 2023](https://doi.org/10.1038/s41467-023-36994-z)) is a
conditional VAE for antimicrobial peptide design. Per the organizing committee's
participation policy, HydrAMP is a **baseline** (excluded from rankings and prizes) and
serves two functions: (1) a published Phase-1 target for participants to beat, and (2) a
sanity check that the evaluation pipeline ranks methods with known experimental outcomes
sensibly.

This repository packages the published HydrAMP inference code and weights behind the
challenge's `generate_*` entry-point contract, so it runs with a single `uv run` command
and emits a validator-compliant library.

## Abstract

HydrAMP is a semi-supervised conditional variational autoencoder for antimicrobial
peptide (AMP) design. It embeds peptides into a continuous latent space that is
disentangled from two conditioning variables (antimicrobial activity and low minimal
inhibitory concentration, MIC), each estimated by an auxiliary classifier. Novel
candidates are produced by sampling latent vectors from the prior, conditioning on
"active" and "low-MIC", and decoding them into sequences; the same auxiliary classifiers
then filter decoded candidates back to the target class. This baseline uses HydrAMP's
default *unconstrained* AMP-generation mode with deterministic (softmax) decoding. On top
of it we apply only automated, deterministic filters: the challenge's hard sequence rules
for the full library, plus HydrAMP's published synthesizability criteria and an 80% novelty
cutoff for the ranked top-100. No sequence optimization, prompt engineering, or manual
curation is performed. Method details: Szymczak et al.,
*Nature Communications* 14, 1453 (2023).

## What this produces

Running this role's standalone entry point from the `vae-latent-models/` directory writes
two files under that directory's `generate_broad_spectrum/`:

```
vae-latent-models/generate_broad_spectrum/
  library.fasta   full 50,000-sequence library
  top.fasta       top-100 ranked candidates
```

These are HydrAMP baseline outputs. They are separate from the canonical root submission
files under the repository-root `generate_broad_spectrum/` and `generate/`. Run this
entrypoint from the role directory because its model, decomposer, and reference defaults are
relative to that working directory.

HydrAMP is strain-agnostic, so only the **broad-spectrum** category is wired up.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/). The pinned model checkpoint is committed under
`checkpoint/`, so no data download is needed.

```bash
cd vae-latent-models
uv run generate_broad_spectrum
```

Optional arguments (all have defaults; the defaults reproduce the published baseline):

| Flag | Default | Description |
|------|---------|-------------|
| `--n-sequences` | `50000` | Number of sequences in the library |
| `--top-k` | `100` | Number of top-ranked sequences |
| `--seed` | `42` | Random seed (fixed for reproducibility) |
| `--model-path` | `checkpoint/model` | HydrAMP checkpoint (epoch 37) |
| `--decomposer-path` | `checkpoint/pca_decomposer.joblib` | Latent PCA decomposer |
| `--antibacterial-fasta` | `data/antibacterial.fasta` | Reference set for the overlap / similarity checks |

Output is byte-identical across repeated runs (deterministic softmax decoding, fixed seeds).

## How it works

`generate_broad_spectrum` calls HydrAMP's `unconstrained_generation(mode="amp")` at default
settings and post-filters to the challenge's hard sequence rules.

The library is generated with HydrAMP's built-in AMP/MIC classifier filtering
(`filter_out=True`, the method's **default**). This yields *designed AMPs* (sequences
HydrAMP predicts to be antimicrobial and active) rather than raw decoder samples, matching
the challenge's request for "50,000 designed AMPs generated under default settings." (Raw
`filter_out=False` output is a deliberate non-default and includes sub-8-residue fragments
and sequences the model itself does not classify as AMPs.)

**Computational filters applied (in order):**

1. *HydrAMP default classifier filtering* (built in): a decoded candidate is kept only if
   its predicted antimicrobial-activity probability and low-MIC probability both fall in
   `[0.8, 1.0]`.
2. *Alphabet*: only the 20 standard amino acids (`ACDEFGHIKLMNPQRSTVWY`).
3. *Length*: 8–50 residues (HydrAMP generates ≤25).
4. *Uniqueness*: global deduplication across the whole library.
5. *No known-antibacterial overlap*: drop any sequence identical to one in
   `data/antibacterial.fasta`.
6. *Top-100 biological synthesizability* (paper's Filtering-1, applied only to the ranked
   list): drop sequences with cysteines, with ≥3 positive residues in any 5-window, with 3
   identical residues in a row, or with 3 identical hydrophobic residues in a row.
7. *Top-100 novelty*: keep only candidates under 80% Levenshtein similarity to every
   sequence in `data/antibacterial.fasta`.

Filters 1–5 shape the 50,000-sequence library; filters 6–7 additionally restrict the
top-100. No hand-selection or manual intervention is applied at any stage.

**Selection procedure (top-100):** candidates are ranked by HydrAMP's `mic` head
(probability of low MIC against *E. coli*; higher is better), and the top-100 is the
highest-ranked subset passing filters 6–7. This reproduces the biological-filtering and
ranking stages of the paper's experimental preselection. The paper's further steps
(consensus of 10 external AMP classifiers and fully-atomistic molecular-dynamics scoring)
are **intentionally not reproduced**: they depend on external tools/web services and
GPU-days of non-deterministic simulation, which are incompatible with the competition's
single-command, fixed-seed reproducibility requirement.

## Model & data provenance

- **Inference code:** [`szczurek-lab/hydramp`](https://github.com/szczurek-lab/hydramp),
  pinned to commit `6590d2f`, installed as a dependency (not modified).
- **Weights:** the `HydrAMP/37` checkpoint and latent PCA decomposer, committed under
  `checkpoint/`, match the official organizer starter kit at commit
  `7804df862872ccc6d09fe01c41bafbca194cfa31`. The checkpoint-tree SHA-256 is
  `02ab8015a717bde799d66cce4043d93e3149a291b8df70c577df4521c76f41d6`; the PCA SHA-256 is
  `a3a5dcea04e91421be363c047249b93fa46e99e128aec929e1f6b0f0e488a26d`. These hashes match
  the assets recorded by the Beam generation run. The public starter-kit commit is the
  retrieval source for the checkpoint.
- **Training data (external databases):** HydrAMP's VAE and its AMP/MIC classifiers were
  trained on publicly available peptide data assembled from:
  - **DBAASP**: Database of Antimicrobial Activity and Structure of Peptides;
  - **DRAMP**: patent, clinical, and general AMP sets;
  - **APD**: Antimicrobial Peptide Database;
  - **GRAMPA**: the aggregated MIC dataset of Witten & Witten
    ([zswitten/Antimicrobial-Peptides](https://github.com/zswitten/Antimicrobial-Peptides));
  - **AMP Scanner v2** ([Veltri et al.](https://www.dveltri.com/ascan/)) AMP/non-AMP set;
  - **UniProt**: source of non-AMP (negative) sequences.

  All sources are public; no proprietary or non-public data was used. The full training set
  is **included in this repository** under [`data/training/`](data/training/) (see
  [`data/README.md`](data/README.md) for a per-file description); the exact assembly and
  preprocessing are in the HydrAMP repository's `scripts/dataset_preparation.ipynb`. Users
  should confirm each source's terms of use for their own purposes.
- **`data/antibacterial.fasta`:** the challenge's reference set of known antibacterial
  peptides, used only for the overlap and similarity checks.
- **Experimental data:** [`experimental/`](experimental/) holds HydrAMP peptides from prior
  wet-lab work with their measured MIC (`mic.csv`) and hemolysis (`hemolysis.csv`) values in
  µM. These are HydrAMP's known experimental outcomes, **not** measurements of the generated
  top-100 (the two sets are disjoint); see [`experimental/README.md`](experimental/README.md).

## Repository layout

Follows the [challenge template](https://github.com/szczurek-lab/amp-challenge-2027):

```
checkpoint/            committed HydrAMP weights (epoch-37 model + PCA decomposer)
data/antibacterial.fasta   challenge reference set for overlap / similarity checks
data/training/         HydrAMP's actual training data (full disclosure)
scripts/verify_submission.py   the challenge validator (run with Python ≥3.9)
src/hydramp_starter_kit/generate.py   the generate_broad_spectrum entry point
experimental/          experimentally characterized peptides + MIC data
metrics/               Phase-1 seqme metrics (added at launch)
pyproject.toml · uv.lock · .python-version
```

## Not yet included

- **`metrics/`**: Phase-1 [`seqme`](https://github.com/szczurek-lab/seqme) metrics for this
  baseline library, computed under the participant protocol, ship at competition launch.

## License

MIT (see [LICENSE](LICENSE)). HydrAMP itself is MIT-licensed.
