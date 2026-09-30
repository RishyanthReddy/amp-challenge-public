# Model generation verification

Two fresh runs of `uv run generate` reproduced the submitted library and ranked Top 100 byte-for-byte on 2026-09-30. This verifies the model generation command, rather than export from saved ranking tables.

## How it was checked

A Linux RTX 4090 worker cloned the public repository at `a0e01a252dba77d89493ee6ba8cc6a2ccb34abe2` and applied the reviewed implementation snapshot. The canonical FASTA folders and saved master, AR and portfolio outputs were removed from the clone before execution. After the first run, its FASTAs and temporary run evidence were removed before the command was run again. Each invocation sampled new candidates, regenerated evolution, calculated fresh forest/APEX predictions, selected the portfolio and assembled the library.

The worker used Beam for compute. The generator itself used no Beam SDK, account, volume or private checkpoint. It downloaded the released checkpoint anonymously and checked its hashes. Dependency environments were reused between runs; generated candidate pools were not reused.

| Check | Run 1 | Run 2 |
| --- | --- | --- |
| Complete command elapsed time | 54.91 min | 50.44 min |
| Initial 3,500 ordered model samples | Original digest matched | Original digest matched |
| Auxiliary A raw/perplexity files | Original hashes matched | Original hashes matched |
| Auxiliary B raw/perplexity files | Original hashes matched | Original hashes matched |
| Eligible AR/evolution pool | 3,964 | 3,964 |
| Ranked Top 100 | 60 AR / 40 evolution | 60 AR / 40 evolution |
| Final FASTAs | Both original hashes matched | Both original hashes matched |

Both runs also passed the exporter and independent FASTA validator: 50,000 unique library members, 100 unique ranked candidates, subset inclusion, the 20-amino-acid alphabet, lengths 8–50, clean headers and no exact library overlap with the 39,448-reference panel. Maximum Top-100/reference Levenshtein ratio was `0.8000`; maximum internal Top-100 ratio was `0.7778`. All 50,000 library sequences passed the synthesis filter.

```text
library.fasta  65183eef34ef86924f311c47de9b9183dabad0634da8586d155c01d81944afba
top.fasta      7181777317b8a4c7c29841471ae9941dfb0de83b1a0c53c5f0a6bf6d0991e480
```

## Runtime and retained inputs

Exact reproduction is scoped to Linux, NVIDIA GeForce RTX 4090, PyTorch 2.5.1 with CUDA 12.4, transformers 4.46.3 and tokenizers 0.20.3, using the locked generation environment. The evaluator and selector use their separate locked environments. Hardware and runtime checks run before model sampling.

The first auxiliary schedule completes 60,000 attempts and retains the historical first 51,069 rows; the second completes and retains 85,000 attempts. This fixed prefix reproduces the original interrupted schedule. It is disclosed in [the generation guide](../generation/README.md).

The 50 disclosed evolutionary seeds and archived HydrAMP/diffusion comparison inputs are retained inputs. The baseline comparison models are not resampled. Their predictions are recomputed, and their ancestry remains excluded from the submitted scored pool. No saved AR/evolution pool, saved APEX prediction table, ranked table or library is used as a generation input.

## Corrections and additional checks

Generic object-key quicksort preserves the original equal-score ordering across ARM and x86. Linux native numeric quicksort had changed the evolutionary trajectory. The corrected ordering reproduces the original ancestry SHA-256, `c6b6e524882951dbb789ecd3e04e99f7c6ab4855feecc7f0c4db513fab9ed2c3`, in the separate Linux diagnostic. APEX also receives an isolated environment so its `uv` invocation cannot replace the live evaluator environment.

The Linux/Mac comparison retained the same eligible pool and ordered Top 100. AMP and hemolysis predictions were unchanged in that comparison; rounded APEX mean MIC differed by at most 0.001 µM. The acceptance result above concerns FASTA identity. It does not claim that new scored Parquet files are byte-identical to the historical Mac files.

Thirteen local regression checks passed in both proposals, including anonymous downloads, hash mismatch rejection, unsupported-host preservation, preservation of both canonical folders after a mismatching regeneration, entry-point mapping, root environment selection, APEX isolation and score ordering. Those local checks are separate from the two actual GPU runs.

## Evidence and limits

- [Complete two-run result and stage manifests](verification/model_generation_20260930/two_run_result.json).
- [Linux selection comparison](verification/model_generation_20260930/linux_selection_result.json), which uses previously regenerated initial samples and is explicitly a diagnostic rather than another full fresh run.
- [Local controls and unchanged artifact checks](verification/model_generation_20260930/local_checks.json).

The released model is loaded rather than retrained by this command. Earlier training and data-curation checks remain in [the original-library replay](ORIGINAL_LIBRARY_REPLAY.md). Model outputs are predictions, not wet-lab measurements. Source-use questions documented in [the source guide](DATA_ACCESS.md) remain separate from computational reproduction; this result does not guarantee co-authorship or competition acceptance.
