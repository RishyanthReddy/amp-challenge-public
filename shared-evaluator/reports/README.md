# Shared evaluator report status

- `training_data_summary.json` and `model_cards.md` document the current classifiers,
  training inputs, grouped hemolysis evaluation, and limitations.
- `master_evaluation_manifest.json` records the current 13,057-row multi-generator input,
  12,756 unique sequences, current evaluator-model hashes, and complete 11-pathogen APEX
  coverage for the 7,655 novelty-eligible, synthesizable ranking candidates.
- `artifact_verification.json`, `master_candidates_audit.md`, and
  `candidate_ingestion_audit.md` describe the same current evaluation; run
  `uv run --frozen python cloud/verify_role6_physical.py` from the repository root to check
  their hashes and invariants.
- Older sign-off documents under `docs/` are historical and are not current evidence.
