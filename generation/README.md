# Reproducing the submitted library

The default `uv run generate` runs the released model and the selection pipeline. It does not read the saved library, saved ranking tables, or saved APEX scores.

## Hardware and commands

Use Linux with an NVIDIA RTX 4090, a driver compatible with CUDA 12.4, and enough disk space for the CUDA dependencies and model files (allow at least 20 GB free for environments, downloads and temporary files). Exact reproduction is scoped to this GPU and the locked software versions. Fixed seeds do not establish identical results on different hardware.

From the repository root:

```bash
uv sync --frozen --python 3.12
uv run generate --check
uv run generate
```

No Beam account or GitHub login is needed. The checkpoint is fetched anonymously from the public release and checked against its recorded hashes. An existing exact checkpoint cache can be reused; candidates are sampled anew on every invocation.

The inference environment uses Python 3.10, PyTorch 2.5.1 with CUDA 12.4, transformers 4.46.3, tokenizers 0.20.3, NumPy 1.26.4 and pandas 2.2.3. The evaluator and selector use their separate locked Python 3.12 environments. The root command installs those environments through `uv`.

## What runs

1. ProGen2 produces the initial 3,500 candidates with seed 42 and the original temperature tiers.
2. Two auxiliary schedules produce 60,000 and 85,000 attempts with base seeds 200,000 and 300,000. The first schedule retains its first 51,069 rows to reproduce the historical interrupted run. Both schedules complete before their retained data is scored.
3. Teacher-forced perplexity is calculated independently; auxiliary candidates above 100 are rejected.
4. Evolutionary search regenerates its candidates and complete ancestry graph from the 50 disclosed seeds, seed 42 and a 3,500-call budget.
5. Reference novelty, biophysical descriptors, the released AMP/RBC forests and the APEX ensemble are applied anew.
6. MAP-Elites annotations and constrained DPP selection produce a ranked Top 100 and nested Top 50. The scored eligible pool and screened auxiliary candidates form the 50,000-member library.
7. The exporter checks sequence constraints and both expected FASTA hashes before replacing any canonical FASTA. An unsuccessful run leaves the submitted files untouched.

The temporary workspace has no saved library or ranked table. Only code, released weights, the challenge reference, seeds and the comparison inputs described below are copied into it. Dependency environments are cached under `.generation-envs/`; successful run evidence is written to `generation_runs/latest_generation.json`. These folders are ignored by Git.

## Historical baseline exclusions

`vae_exclusions.csv` and `diffusion_exclusions.csv` retain a minimal record of the earlier baseline comparison pools: sequence, source identifier, model name and novelty fields. Their hashes, counts and original export hashes are in `comparison_pool_manifest.json`.

These are generated comparison outputs, not training data or submitted peptides. Original model names and source identifiers are retained as attribution; this inclusion does not grant additional rights in the baseline software or weights. They preserve the original rule excluding sequences with any HydrAMP/diffusion ancestry, including cross-domain duplicates. Including them during fresh scoring also preserves the historical ordering of equal-score candidates. Their model scores are recomputed; no saved predictions are read. Baseline models are not rerun. Every scored comparison row with either baseline's ancestry is excluded before portfolio selection. Auxiliary library sequences are sampled independently from ProGen2 and retain their own source records.

The evolutionary seed panel is included with its original source metadata and attribution. All 50 peptides occur in the bundled DBAASP FASTA. Reference-matching seeds are rejected as submission candidates; mutations remain subject to the same screening.

## Ordering and numerical scope

Evolutionary survival and the master candidate order use generic object-key quicksort for numeric scores. This retains the original equal-score ordering without changing score values or dataframe dtypes. Native numeric quicksort produced a different mutation trajectory on Linux x86 than on the original Mac ARM run. The generic ordering reproduces the original ancestry graph on both platforms.

APEX runs in its own environment rather than inheriting the caller's `uv` environment. APEX mean-MIC values differed by at most 0.001 µM after rounding in the Linux/Mac comparison; the verified Linux selection retains the same eligible candidates and ordered Top 100. Exact FASTA hashes are the final acceptance check.

## CPU inspection and export

To inspect the unchanged submitted artifacts without a GPU:

```bash
uv run export_submission
uv run python scripts/verify_submission.py . --no-replay
```

`export_submission` exports the saved selection tables. It is explicitly separate from model generation. It does not establish that models have run.

The full validator without `--no-replay` reruns the default model generator and compares it with the existing FASTAs, so it requires the declared GPU and time for a complete run.
