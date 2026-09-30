# Replaying the research method

The root `uv run generate` now runs the released model and the selection pipeline on the declared RTX 4090/runtime. See [the generation guide](../generation/README.md). `uv run export_submission` is the separate CPU exporter. The training replay and historical evidence below remain useful for examining how the released checkpoint and original datasets were produced.

## 1. Work in an isolated checkout

Research scripts write to component output directories. Make a separate checkout before running them. Use a separate Beam volume for new training and unique run IDs for sampling. Never use `--promote-canonical` for a replay.

Restore the public sources described in [DATA_ACCESS.md](DATA_ACCESS.md). The exact retained DRAMP General files are included with attribution because current downloads differ. The archived serialized views are listed in [replay_inputs.json](replay_inputs.json). They remain in the private archive; fresh curation now recovers the same ordered model-input values from the restored sources, as verified below. Source terms remain applicable.

```bash
uv run python scripts/fetch_training_sources.py
uv run python scripts/fetch_training_sources.py --verify-only
uv run python scripts/verify_replay_inputs.py
uv run --project shared-evaluator --locked --python 3.12 python \
  scripts/verify_evaluator_assets.py --smoke
```

The all-source check passes after pinned downloads. `verify_replay_inputs.py` checks archived serialized bytes; it requires the original curated files. For a fresh curation rebuild, use the value check below instead. Parquet writer versions change file bytes without changing model inputs. The public forests can be used directly without fitting new ones.

## Rebuild curated views from the restored sources

Run this in the isolated checkout:

```bash
uv run --project data-engineering --locked --python 3.12 amp-data-build
uv run python scripts/verify_rebuilt_views.py
uv run --project shared-evaluator --locked --python 3.12 python \
  shared-evaluator/scripts/train_evaluator_models.py
uv run python scripts/verify_evaluator_assets.py
```

The value check matches the original columns, dtypes, row order and exact scalar values.
Only `source_file` path separators are normalized. The 2026-09-29 rebuild and refit
recovered both original model files exactly; their serialized input tables differ because
the originals used Arrow 23.0.0 and the locked data environment uses Arrow 25.0.1.

## 2. Train and produce the scored candidate pool

The earlier isolated replay used the original training mathematics in a recorded Beam runner: 26,699 training sequences, three epochs, seed 42, and validation-selected epoch 1. The checkpoint reproduced exactly. Executed runner snapshots and input fingerprints remain in the private archive under `docs/verification/full_replay_20260928/`. The local fine-tuning CLI is also available:

```bash
uv run --project autoregressive-models --locked python \
  autoregressive-models/scripts/05_finetune.py \
  --views-dir data-engineering/data/processed/views \
  --epochs 3 --batch-size 8 --lr 0.00005 --seed 42
```

That CLI is an alternative execution path; its own output must be compared rather than assumed byte-identical to the historical Beam checkpoint. Sampling, candidate audit, evaluator training, evolution and APEX scoring use the entry points in [running.md](running.md). The original 3,500 AR candidates, their audit and the 3,500 evolutionary candidates were previously reproduced. The ancestry file reproduced byte-for-byte. APEX and MAP-Elites/DPP recovered the original ranked Top 100 and nested Top 50.

For exact historical selection, retain the four original audited input tables and their master-evaluation manifest. HydrAMP and diffusion rows are used here only to reproduce the historical source-exclusion decisions; no sequences with either baseline's ancestry enter the submitted portfolio. Replacing them with empty files can alter duplicate attribution and therefore the eligible pool.

## 3. Reproduce the original auxiliary schedule

[original_auxiliary_schedule.json](original_auxiliary_schedule.json) records the schedule and expected hashes. The executed sampler is `autoregressive-models/outputs/progen_aux_production_runner_20260928.py`; its original source hash is `22045b357c2fdc284bbdc331a70e2ebb41db767393cd7f8e0b964a6ff961efd1`. The executed teacher-forced scorer is `progen_aux_perplexity_runner_26a039f20968.py`.

Use Python 3.10, PyTorch 2.5.1, transformers 4.46.3, tokenizers 0.20.3 and RTX 4090, with the pinned checkpoint/tokenizer. Copy only the runner into a small upload directory. Change its volume name to an isolated volume containing the verified checkpoint; record that source change and its hash.

| Run | Requested attempts | Historical rows used | Base seed | Batch size |
| --- | ---: | ---: | ---: | ---: |
| A | 60,000 | 51,069 | 200000 | 64 |
| B | 85,000 | 85,000 | 300000 | 64 |

Run A was interrupted historically. To reproduce its retained prefix, complete the original 60,000-attempt schedule, then take its first 51,069 ordered rows. Requesting only 51,069 attempts changes the final batch size and can change sampling. Run B should complete all 85,000 attempts. Use the original temperature cycle (0.85, 1.00, 1.15), top-p 0.90 and lengths 8–50. Compare every original raw CSV column, then independently score the retained attempts at perplexity <=100.

The original `run_progen_auxiliary.py` entry point is a later orchestration version; the archived executed sampler is the authority for exact historical mathematics. Its local generation manifest must be supplemented with the archived runner hash before passing it to the PPL runner. Record the completed remote result and any deliberately retained prefix honestly. Do not label the original interrupted run completed.

## 4. Select and assemble in isolation

With a verified master table and valid generation/PPL manifests:

```bash
uv run python scripts/build_submission_artifacts.py \
  --run-id my_isolated_replay \
  --aux-csv autoregressive-models/outputs/my_run_a_ppl.csv \
  --aux-csv autoregressive-models/outputs/my_run_b_ppl.csv
```

This creates a challenger directory, validates the portfolio, scores the auxiliary pool, writes FASTAs and checks submission invariants. Source-table hashes, runner hashes and raw/PPL manifests are mandatory. When comparing the existing submission byte-for-byte, preserve the original sequence IDs as well as sequence order. New run labels change FASTA identifiers even when the peptides match.

Compare library and ranking order, then the FASTA hashes against [FINAL_HANDOFF_MANIFEST.json](FINAL_HANDOFF_MANIFEST.json). Keep replay files separate. Successful generation of a different valid 50,000-member library is useful evidence, but does not prove regeneration of the submitted one.

## Evidence and remaining limits

[The historical report](FULL_REPLAY_VERIFICATION.md) records the previous training, scoring and selection replay. The [2026-09-29 original-library replay](ORIGINAL_LIBRARY_REPLAY.md) now records byte-identical sampling/PPL and final artifacts, plus raw curation and evaluator refitting. The completion checklist records outstanding publication and source-use questions. No computational replay establishes biological efficacy or guarantees co-authorship.
