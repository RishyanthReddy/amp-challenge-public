# Data release notes

## Version 1.0 - 2026-08-24

Initial reproducible data-engineering release. Includes immutable raw-file manifest, canonical sequences, provenance links, assay tables, exact official-reference overlap report, similarity clusters, leakage-safe novelty and activity splits, role-specific views, data card, QC reports, and rebuild instructions.

Known limitations: APD/APD3, HydrAMP, and external controlled negatives are not local inputs. Condition-specific DBAASP inactivity notes are preserved as `tested_inactive` assay evidence, but there is no global non-AMP negative class. Reference reporting is exact-match only; nearest-neighbour similarity is not calculated in version 1.0. Evolution seeds are diverse cluster representatives but are not asserted to be natural without evidence. No compatibility-based repeated-assay summary table is included; consumers must use the full observation table.

Any correction must create a patch release with affected artifacts and model runs documented.
