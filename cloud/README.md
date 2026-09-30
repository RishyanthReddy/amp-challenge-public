# GPU runners and model downloads

These scripts support training, model inference, asset retrieval and artifact verification.
The root `uv run generate` command runs the models locally on the declared GPU; it does not dispatch Beam jobs.
`uv run export_submission` is the separate CPU artifact export.

## Checkpoint access

```bash
uv run python cloud/fetch_progen_checkpoint.py --source public
```

The `public` mode downloads the checkpoint release anonymously. The default `github` mode
uses GitHub CLI, which may request login; browser downloads also work without authentication.
`--source beam` uses the original model volume instead. Every file is checked against size
and SHA-256 pins. `fetch_diffusion_checkpoint.py` downloads the separately pinned baseline.

## Beam runners

| Script | Function |
| --- | --- |
| `run_ar_perfect.py` | ProGen2 training, validation checkpoint selection and initial sampling |
| `sample_from_volume.py` | Batched inference from an existing ProGen2 checkpoint |
| `run_progen_auxiliary.py` | Bounded auxiliary generation with run manifests |
| `score_progen_auxiliary.py` | Teacher-forced perplexity screening |
| `run_hydramp_pipeline.py` | HydrAMP baseline experiments |
| `run_diffusion_pipeline.py` | Diffusion baseline experiments |

These require the Beam Python SDK, authenticated Beam CLI and access to the configured
volume. Run from a small dedicated upload directory containing the necessary runner files;
use `AMP_CHALLENGE_ROOT` where supported to locate the local project outputs.
Do not upload the entire research workspace as a cloud function bundle.

Historical training scripts save into their configured model volume and output paths.
Use a separate checkout and model volume for new experiments. The executed isolated
verification sources are preserved in [replay summary](../docs/FULL_REPLAY_VERIFICATION.md); detailed runner records remain in the private archive.

## Candidate audits and verification

`audit_ar_candidates.py`, `audit_vae_candidates.py`, and `audit_diffusion_candidates.py`
check candidates and record novelty/synthesis fields before integration. Physical verifiers
check individual experiment manifests; some require raw research outputs that are not
included in a minimal checkout.

For the submitted artifact check, use:

```bash
uv run python scripts/verify_submission.py .
```

See [execution guide](../docs/running.md) and [verification results](../docs/FULL_REPLAY_VERIFICATION.md).
