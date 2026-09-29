# Dataset schema, version 1.0

## Canonical sequences

`processed/sequences.parquet` has one row per normalized exact sequence. `sequence_id` is a deterministic SHA-256-derived identifier. Sequence validity, modification, stereochemistry, challenge-reference, and eligibility fields are explicit.

## Source links

`processed/provenance.parquet` is the source-link table. It preserves `source_database`, `source_file`, `source_record_id`, original header, original sequence, normalized sequence, and source metadata for every input appearance.

## Experimental measurements

`processed/activity.parquet` stores one activity observation per source assay row. `processed/toxicity.parquet` separately stores hemolysis/cytotoxicity observations. Measurements are never collapsed in these master tables.

MIC fields preserve `mic_value_raw`, `mic_unit_raw`, `mic_qualifier`, numeric/bound fields, conversion status, molecular weight, and µM values only where valid.

## Labels and evidence

`annotations.parquet` supplies source-derived AMP, antibacterial, antibiofilm, hemolysis, and cytotoxicity indicators. `activity.parquet` also has condition-specific `activity_evidence_label`: `tested_inactive` when a source assay note explicitly reports inactivity, `experimental_measurement_reported`, or `unknown`. Missing observation is unknown, never negative. A condition-specific inactive result is not promoted to a global non-AMP label. The evaluator view therefore uses `negative_label = unknown_no_global_negative_control_source`.

## Splits and views

`novelty_safe_*` views are reference-filtered and similarity-clustered. `train.parquet` is the complete validated known-AMP corpus. `activity_prediction/*` uses a separate MIC-observation 80%-cluster split. Role-specific views are in `processed/views/`.
