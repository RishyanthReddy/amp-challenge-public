# Portfolio report status

- `portfolio_run_manifest.json` records the canonical AR/evolution selector inputs, source
  hashes, Top 50/Top 100 Parquet hashes, and domain distribution.
- `library_assembly_manifest.json` records the 50,000-row library, selected ProGen2 auxiliary
  runs, raw and PPL-scored file hashes, source-runner hashes, synthesis counts, and reference
  checks.
- `submission_entry_manifest.json` joins the canonical portfolio and assembly manifests,
  exact output hashes, and the conservative method interpretation.
- `submission_artifact_verification.json` is refreshed by
  `cloud/verify_role7_physical.py`, which checks source tables, manifests, auxiliary PPL
  decisions, Parquet-to-FASTA order, synthesis passage, and canonical hashes.
- `portfolio_lottery_report.md` compares exact hypergeometric probabilities against a seeded
  Monte Carlo simulation of model-score labels. It is not an estimate of wet-lab hit rates.

Official HydrAMP and AMP-Diffusion checkpoint outputs are excluded from the current ranked
portfolio. The integrated ProGen2/evolutionary entry classification and data/model release
rights remain open; this report does not certify competition eligibility.
