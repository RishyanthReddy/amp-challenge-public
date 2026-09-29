# Checkpoint access and repeatable inference

The fine-tuned ProGen2 checkpoint is available as four assets in the
[public checkpoint release](https://github.com/RishyanthReddy/amp-challenge-public/releases/tag/progen2-checkpoint-20260928).
All four files were downloaded without authentication and checked against their expected
sizes and SHA-256 hashes. No Beam access is needed for this download.

From the repository root, with GitHub CLI installed:

```bash
uv run python cloud/fetch_progen_checkpoint.py
uv run python scripts/verify_inference_replay.py --output outputs/inference_check_01
```

GitHub CLI may request login; the release also supports direct browser downloads. The
fetcher refuses an unexpected cached file and installs downloaded files only after their
hashes match. To use a browser download, place `config.json`, `generation_config.json`,
`tokenizer.json` and `model.safetensors` in
`autoregressive-models/checkpoints/progen2_small_amp_best_val/`, then run the fetcher to
check them locally.

The inference check installs the locked autoregressive environment, loads the pinned model
code and generates one candidate twice on CPU with seed 42. It checks finite perplexity
at or below 100 and compares output hashes. Use a fresh `--output` directory for each run.
The recorded check produced identical outputs; see
[inference_replay_verification.json](inference_replay_verification.json).

A one-candidate repeatability check does not establish byte identity across CPU and GPU
hardware or reproduce the entire submitted library. For larger samples, use the
[autoregressive CLI](../autoregressive-models/README.md) with an isolated output directory.
The [historical replay report](FULL_REPLAY_VERIFICATION.md) describes the separate training,
evolution, scoring and selection replay performed with archived research inputs.

The root `uv run generate` remains the deterministic export of frozen selected tables.
