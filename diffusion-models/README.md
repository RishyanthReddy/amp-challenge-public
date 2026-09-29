# AMP-Diffusion research baseline

This component contains the AMP-Diffusion starter-kit model, an ESM-2 decoder, a bounded
generation adapter, and a separate APEX scoring environment. It is not called by the root
submission packager.

## Source and generation

- `src/ampdiffusion_starter_kit/model.py` defines `Denoise_Transformer` and
  `GaussianDiffusion1D`.
- `scripts/generate.py` is the role adapter for 1000-step DDPM or 250-step DDIM. It validates
  arguments, records a maximum-round bound, and filters unique valid outputs.
- `cloud/run_diffusion_pipeline.py` is the production Beam runner. It requires CUDA and
  records per-batch seeds, model/checkpoint hashes, ESM asset hashes, sampler settings, and
  runtime versions.
- `cloud/audit_diffusion_candidates.py` quarantines exact challenge-reference matches,
  recomputes full-precision Indel novelty, and adds biophysical and synthesizability fields.

## Current run

On 2026-09-27 the cloud runner completed on an NVIDIA RTX 4090. It generated 100 smoke
sequences at 1000 steps, compared 500 candidates at 250-step DDIM and 1000-step DDPM, and
generated 3,500 production candidates with 1000-step DDPM. The audit quarantined 5 exact
reference matches, retaining 3,495 unique sequences; 3,360 pass the `<=0.80` novelty check
and 3,101 pass the synthesizability screen.

The 500-sequence route comparison recorded 250-step DDIM at 60.38 seconds (8.28 sequences/s)
and 1000-step DDPM at 239.02 seconds (2.09 sequences/s). These timings describe this Beam
run and hardware, not a biological-quality comparison. See
`reports/feasibility_metrics.csv`, `reports/step_tradeoff.csv`,
`reports/diffusion_generation_manifest.json`, and
`reports/diffusion_candidate_audit_manifest.json`.

## Weights and replay

The production checkpoint SHA-256 is
`6a3f347df7c02ff6008ac3d2d4826daeadf7418cb6862a6599b4f710e1d7f8aa`. It was mounted from
Beam's `amp-models` volume for the recorded production run. A fresh clone can fetch the same
released checkpoint with the hash-pinned helper:

```bash
uv run --frozen python cloud/fetch_diffusion_checkpoint.py
uv run --project diffusion-models --locked \
  python diffusion-models/scripts/generate.py --n-sequences 100
```

The helper downloads `checkpoint/model.pt` from official starter-kit commit
`1a862af9078e6b55c87d1fa576f3da81851ba94b`, then checks the exact 132,526,179-byte size and
SHA-256 before atomically placing the file at the ignored local cache
`cloud/diffusion_checkpoint/model.pt`. The standalone adapter uses that same path by default.
The upstream starter kit is MIT-licensed and stores the checkpoint in Git LFS. ESM-2 8M
weights and the regression asset are fetched by `fair-esm` and their hashes are recorded in
the generation manifest. APEX model assets are tracked separately and still need a
redistribution-terms review.

## Baseline eligibility and scientific limits

The [AMP-Diffusion starter-kit README](https://github.com/szczurek-lab/ampdiffusion-starter-kit)
states its baseline outputs are excluded from rankings. Under the conservative local method
decision, AMP-Diffusion candidates are retained as research artifacts but excluded from the
current ranked portfolio and library. A separately trained checkpoint could change that
classification if its provenance is established. All activity, safety, and MIC values are
model predictions. No generated candidate has been assayed by this project.
