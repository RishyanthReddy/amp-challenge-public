# Example rows

These illustrative shapes are derived from the generated schemas; identifiers and sequences are placeholders.

```text
sequences: sequence_id=seq_abcd..., sequence=ACDEFGHI, length=8,
           modification_status=none_known, stereochemistry_status=l_or_unspecified,
           challenge_exact_overlap=false

provenance: sequence_id=seq_abcd..., source_database=DBAASP_peptides,
            source_record_id=123, original_sequence=ACDEFGHI,
            sequence_normalized=ACDEFGHI

activity: sequence_id=seq_abcd..., target_organism=Escherichia coli ATCC 25922,
          assay_type=MIC, mic_value_raw=< 4, mic_qualifier=<,
          mic_lower_bound=4.0, mic_value_uM=null
```

`null` for an exact µM value means the observation is censored or ranged, not missing processing.
