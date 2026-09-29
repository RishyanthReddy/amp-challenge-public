# data/

Two distinct things live here; do not confuse them.

## `antibacterial.fasta`: challenge reference set (NOT training data)

The **AMP Challenge 2027 reference set** of known antibacterial peptides, copied verbatim from
the [challenge template](https://github.com/szczurek-lab/amp-challenge-2027). Used only by the
generator/validator to enforce two challenge rules:

- no library sequence may be **identical** to any sequence here;
- no top-100 sequence may exceed **80% Levenshtein similarity** to any sequence here.

## `training/`: AMP-Diffusion's training data

The AMP-Diffusion checkpoint shipped in `checkpoint/model.pt` (the released 16.54M-parameter
ESM2-8M variant) was trained on a compiled set of antimicrobial peptides drawn from three
public databases, per Torres et al., *Cell Biomaterials* 2025 (Dataset S1):

| Source | License / access |
|--------|------------------|
| **DRAMP 3.0** — Data Repository of Antimicrobial Peptides | public, academic use |
| **APD3** — Antimicrobial Peptide Database | public |
| **DBAASP** — Database of Antimicrobial Activity and Structure of Peptides | public (CC BY-like) |

Sequences were compiled and de-duplicated; the model operates on ESM2 embeddings of these
AMPs. The *Cell Biomaterials* paper is **open access under CC BY 4.0**, so the training set is
redistributed here with attribution.

`training/training.fasta` — the **19,670 unique** training sequences, taken verbatim from the
AMP-Diffusion release (`train_eval.csv`). All standard amino acids; lengths 1–50.

The earlier bioRxiv AMP-Diffusion (Chen et al. 2023) used a larger 195,121-sequence set from
dbAMP, AMP Scanner, and DRAMP; the checkpoint shipped here is the *Cell Biomaterials* model,
whose training set is the one described above.

**Provenance / terms.** All sources are public. Users should confirm each upstream database's
terms of use for their own purposes.
