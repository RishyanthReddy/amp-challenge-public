# Autoregressive peptide generation

ProGen2-small fine-tuning, seeded amino-acid sampling, and sequence-level perplexity
screening. The fine-tuned checkpoint is available through the private GitHub release.

## Sample candidates

From the repository root, with GitHub CLI authenticated:

```bash
uv run python cloud/fetch_progen_checkpoint.py
uv run --project autoregressive-models --locked python \
  autoregressive-models/scripts/06_generate_from_checkpoint.py \
  --checkpoint autoregressive-models/checkpoints/progen2_small_amp_best_val \
  --n 100 --seed 42 --max-perplexity 100 \
  --output-dir outputs/inference_example
```

CUDA is selected when available; otherwise inference runs on CPU. `--n` counts generation
attempts. Filtering can retain fewer rows, and sampling alone does not perform final
reference-novelty checks or portfolio selection.

## Source map

| Path | Purpose |
| --- | --- |
| `common/model_utils.py` | Pinned model loading, amino-acid token masking, cached sampling and perplexity |
| `scripts/05_finetune.py` | Fine-tuning from curated data partitions |
| `scripts/06_generate_from_checkpoint.py` | Checkpoint inference and optional perplexity screening |
| `docs/` | Candidate and training analysis |
| `outputs/` | Research results and generation provenance |

The component uses its own `pyproject.toml` and `uv.lock` because the pinned ProGen2 custom
code requires Transformers 4. The model-code revision is recorded in `common/model_utils.py`.
Checkpoint files are hash-verified and excluded from Git; see [model assets](../docs/assets.md).

For full workflow instructions, see [execution](../docs/running.md). The submitted checkpoint
and ranked selection were reproduced in the [full replay](../docs/FULL_REPLAY_VERIFICATION.md).
