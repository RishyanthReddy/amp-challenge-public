# Portfolio selection

This component contains MAP-Elites, constrained DPP selection, library assembly, and portfolio
analysis. Python dependencies are pinned in `pyproject.toml` and `uv.lock`.

## Export or repeat selection

The public checkout includes frozen library and Top-100 tables for deterministic export.
Fresh selection needs the scored master table and auxiliary generation inputs retained in
the private research archive. The workflow below describes that research path; see the
[execution guide](../docs/running.md) for which commands can run from the public checkout.

## Ranked-entry workflow

1. `scripts/run_submission_portfolio.py` verifies the shared-evaluator input and manifest hashes,
   excludes any candidate with HydrAMP or AMP-Diffusion ancestry, and requires validity,
   synthesis, `<= 0.80` reference novelty, and positive APEX mean MIC. It writes an isolated,
   nested Top 50/Top 100 plus reserves and a manifest.
2. `scripts/assemble_submission_library.py` verifies the raw ProGen2 and PPL-scored auxiliary
   run manifests and their source snapshots. It assembles the 3,964 scored AR/evolution
   candidates with 46,036 auxiliary ProGen2 fillers, checks exact-reference exclusion, and
   applies the synthesis filter to every selected library sequence.
3. `scripts/build_submission_artifacts.py` packages an isolated challenger and verifies it.
   `--promote-canonical` also archives the previous four-domain research snapshot before
   updating local canonical tables and FASTAs. It performs no remote writes.
4. The root `uv run generate` and `uv run generate_broad_spectrum` commands run model inference,
   mutation, fresh scoring, selection and assembly on the declared RTX 4090/runtime.
   `uv run export_submission` is the separate CPU export of saved tables.
   `scripts/verify_submission.py` checks FASTA invariants and can repeat model generation.

The current canonical run ID is `final_ar_evo_20260928`. The Top 100 has 60 autoregressive
and 40 evolutionary candidates; all 50,000 library sequences are unique and pass the
synthesis filter. The auxiliary sequences were PPL-screened at `<= 100` and are not APEX MIC
scored. Detailed counts and hashes are in
`reports/submission_entry_manifest.json` and
`reports/submission_artifact_verification.json`.

`scripts/run_production_portfolio.py` and `scripts/audit_inputs_and_assemble_library.py`
support the earlier four-domain research workflow. The current submission-entry manifest
guards canonical outputs against accidental replacement by that historical route.
`scripts/package_submission.py` remains a compatibility wrapper to the root packager.
The public checkout contains the canonical root FASTAs and their `generate_broad_spectrum/`
compatibility copies.

## Selection and scoring limits

The DPP combines an RBF kernel over standardized biophysical features and the cosine Gram
matrix of normalized amino-acid compositions. Their weighted sum is PSD; diagonal quality
scaling preserves PSD, and positive diagonal jitter supports Cholesky factorization. The
current selector fails closed on novelty and pairwise cosine/Levenshtein redundancy
constraints and enforces representation of both entry domains. The cosine threshold alone
does not imply a sequence-level Levenshtein bound; the explicit sequence guard and final
verifier enforce `<= 0.80`.

MAP-Elites uses six charge bins, five hydrophobic-moment bins, and five length bins (150
cells). The LCB-style quality score penalizes ensemble spread and toxicity predictions; it
is not a statistically calibrated confidence bound. The lottery report compares exact
hypergeometric probabilities with a seeded Monte Carlo model-score simulation; neither
predicts wet-lab hit rates.

## Entry scope

The selected entry uses ProGen2 and evolutionary search. HydrAMP and AMP-Diffusion baseline
outputs are excluded. See the [method](../docs/method.md),
[execution guide](../docs/running.md) and [verification report](../docs/FULL_REPLAY_VERIFICATION.md).
Canonical submission FASTAs are at the repository root under `generate/`.
