# Portfolio records

| Record | Contents |
| --- | --- |
| `portfolio_run_manifest.json` | Final AR/evolution selector inputs, source hashes and domain distribution |
| `library_assembly_manifest.json` | Library assembly inputs, auxiliary run hashes, synthesis counts and reference checks |
| `submission_entry_manifest.json` | Final portfolio and assembly records with output hashes |
| `submission_artifact_verification.json` | Recorded checks of source tables, FASTA order, novelty and synthesis |
| `portfolio_lottery_report.md` | Hypergeometric and Monte Carlo calculations using model-score labels |
| `selection_decision.md` | Earlier four-generator comparison; it is not the final two-generator selection |

The lottery calculations do not predict wet-lab hit rates. Physical verifiers require
research tables and auxiliary run files held in the private archive. For the included
FASTAs, run `uv run python scripts/verify_submission.py .` from the repository root.
See [research-record guidance](../../docs/research_records.md).
