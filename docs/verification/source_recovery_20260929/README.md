# Source recovery evidence

Checks performed on 2026-09-29 against retained project inputs and public downloads. Workspace paths were made relative; scientific counts and fingerprints were not changed.

- `source_manifest_verification.json`: all 11 raw source files match the historical manifest.
- `source_comparison.json`: current DRAMP and pinned UniProt file comparisons.
- `dramp_general_record_comparison.json`: retained/current general DRAMP records.
- `dbamp_snapshot_match.json`: byte-identical public research archive.
- `training_source_coverage.json`: overlap-aware training-source coverage.

The comparison date is not an acquisition date. A match proves content identity, not licensing permission or original acquisition provenance. See [source access](../../DATA_ACCESS.md).

## Implementation checks

The downloader's 11 offline regression checks pass. Real downloads verify 12 bundled or
retrievable inputs in the initial public check. After preparing the two retained DRAMP General files, all 14 inputs verify in the public checkout. The private checkout verifies all 14 retained
inputs and ten curated replay inputs. Both original forests load in their locked environment,
and the public APEX wrapper returns finite positive predictions for all 11 pathogens.
Submission verification passes in both checkouts, including repeated byte-identical export.
See the adjacent JSON evidence.

```bash
uv run python scripts/verify_training_source_fetch.py
```
