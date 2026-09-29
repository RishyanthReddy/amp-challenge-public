# Baseline data requirements

The public checkout does not bundle AMP-Diffusion's training corpus. It retains baseline
source code and generation manifests; the baseline checkpoint is obtained with the
[hash-pinned download helper](../../cloud/fetch_diffusion_checkpoint.py).

The challenge reference is included at
[`data-engineering/data/challenge/antibacterial.fasta`](../../data-engineering/data/challenge/antibacterial.fasta).
It is used for novelty validation and exact-match exclusion, not as this project's declared
training source.

Historical diffusion records refer to the upstream checkpoint's training data. They do not
establish that training files are distributed in this checkout. For upstream model and data
provenance, consult the [AMP-Diffusion starter kit](https://github.com/szczurek-lab/ampdiffusion-starter-kit)
and its linked publication. Source terms apply independently of this project's MIT license.

No AMP-Diffusion-generated sequence is included in the selected submission.
