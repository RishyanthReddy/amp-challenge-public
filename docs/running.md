# Execution guide

Run commands from the repository root. Use `uv` to select each component's locked
environment; the root environment is intended for FASTA export and validation.

## 1. Export the submitted sequences

```bash
uv sync --frozen --python 3.12
uv run generate
uv run python scripts/verify_submission.py .
```

The command reads the frozen library and ranked tables in `portfolio-selection/outputs/`
and writes `generate/library.fasta` and `generate/top.fasta`, with identical compatibility
copies under `generate_broad_spectrum/`. It requires no GPU or model download. The validator
checks counts, uniqueness, alphabet, lengths, subset membership, novelty and repeated export.

## 2. Download weights and sample new sequences

The release assets are public. GitHub CLI may require login; direct browser downloads do not.

```bash
uv run python cloud/fetch_progen_checkpoint.py
uv run --project autoregressive-models --locked python \
  autoregressive-models/scripts/06_generate_from_checkpoint.py \
  --checkpoint autoregressive-models/checkpoints/progen2_small_amp_best_val \
  --n 100 --seed 42 --max-perplexity 100 \
  --output-dir outputs/inference_example
```

`--n` is the number of generation attempts, not a guarantee of that many retained unique
sequences. Output CSVs contain sequence, validity, perplexity and seed metadata. Set
`--device cpu` or `--device cuda` explicitly if needed. These outputs are separate from the
submitted portfolio. The fetcher refuses mismatching cached weights.

For a small repeatability check:

```bash
uv run python scripts/verify_inference_replay.py --output outputs/inference_check_01
```

Use a new output directory for each check. The check downloads verified weights, generates
one candidate twice on CPU and compares results. It is distinct from the full replay
recorded in [the verification report](FULL_REPLAY_VERIFICATION.md).

## 3. Train, evaluate and select research candidates

Additional inputs are required: curated training tables, AMP/RBC forests and baseline
checkpoints are omitted. These commands document the research workflow and cannot run
from this candidate alone. See [source access](DATA_ACCESS.md).

These stages write research outputs. Run them in an isolated checkout if preserving a
previous candidate pool. Full regeneration needs curated training partitions, all evaluator
assets, and a GPU for practical ProGen2 training/sampling throughput.

| Stage | Entry point | Environment / output |
| --- | --- | --- |
| Data curation | `data-engineering/src/amp_data/build.py` | Data component; processed views and provenance |
| ProGen2 training | `autoregressive-models/scripts/05_finetune.py` | AR component; checkpoint and training metrics; use `--help` for options |
| GPU training and sampling | `cloud/run_ar_perfect.py`, `cloud/sample_from_volume.py` | Beam SDK; remote model volume and candidate CSVs |
| Auxiliary sampling / PPL | `cloud/run_progen_auxiliary.py`, `cloud/score_progen_auxiliary.py` | Beam SDK; uniquely named CSVs and hash manifests |
| Evolution | `evolutionary-search/scripts/run_production_search.py` | Evolution component; candidates and ancestry |
| Evaluator training | `shared-evaluator/scripts/train_evaluator_models.py` | Evaluator component; AMP and RBC forests |
| Candidate scoring | `shared-evaluator/scripts/run_master_evaluation.py` | Evaluator plus isolated APEX environment; scored master table |
| Selection and assembly | `scripts/build_submission_artifacts.py` | Root orchestrator; isolated challenger outputs |

For example:

```bash
uv run --project shared-evaluator --locked python shared-evaluator/scripts/train_evaluator_models.py
uv run --project evolutionary-search --locked python evolutionary-search/scripts/run_production_search.py
uv run --project evolutionary-search --locked python evolutionary-search/scripts/audit_evolution_candidates.py
```

Candidate tables must have the audit fields expected by the shared evaluator. Audit AR
candidates with `cloud/audit_ar_candidates.py` before scoring. The complete pipeline is a
sequence of component commands; `uv run generate` is not an end-to-end training command.

To assemble from already generated, PPL-scored auxiliary inputs and a current scored master:

```bash
uv run python scripts/build_submission_artifacts.py \
  --run-id my_experiment \
  --aux-csv autoregressive-models/outputs/my_auxiliary_ppl.csv
```

Each auxiliary CSV needs its matching generation, scoring and source-hash manifests. The
assembler validates the pinned checkpoint and provenance, and refuses an existing run ID.
Results go to `portfolio-selection/outputs/challenger_my_experiment/`; the submitted files
are only replaced if the explicit `--promote-canonical` option is supplied.

Beam runners use named persistent volumes. The historical training runner writes its
checkpoint to that volume; use a separate volume for a verification or retraining run.
The executed isolated runner snapshots and input hashes are retained with the
[full replay evidence](FULL_REPLAY_VERIFICATION.md).
