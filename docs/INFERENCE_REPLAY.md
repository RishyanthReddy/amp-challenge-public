> Full selected training-to-selection replay: **PASS**. See [FULL_REPLAY_VERIFICATION.md](FULL_REPLAY_VERIFICATION.md) for current evidence and scope.

# Checkpoint access and inference replay

The existing ProGen2 research checkpoint is stored as four assets on the private GitHub
prerelease `progen2-checkpoint-20260928`. It can be downloaded by a repository collaborator
without Beam access. No organizer invitations have been sent. This access route does not
resolve the source-data permissions in `training_source_scope.json`.

From a fresh clone with `gh` authenticated to an account with repository access:

```sh
uv run python cloud/fetch_progen_checkpoint.py
uv run python scripts/verify_inference_replay.py
```

The fetcher checks committed byte sizes and SHA-256 pins, refuses an unexpected local file,
and installs downloaded files only after validation. The inference check installs the locked
Role 02 environment, loads the pinned remote model code, generates one candidate twice on
CPU with seed 42, validates finite perplexity <=100, and compares output hashes. Results go
to `outputs/inference_replay_check/report.json`. Use a fresh `--output` directory for a rerun.
The check needs network access for GitHub, Python packages, and pinned Hugging Face model code.

For Beam access instead, use `--source beam`. The original Beam assets remain available.
For larger sampling, use Role 02's CLI with `--n`, `--seed`, `--device cuda`, and an isolated
`--output-dir`. An inference smoke check is not a rerun of the 50,000-sequence production
pipeline and does not establish byte identity across CPU and GPU hardware.

The root `uv run generate` remains deterministic packaging of frozen results. Full replay
of candidate generation, evolution, evaluator scoring, and selection is still separate work.
