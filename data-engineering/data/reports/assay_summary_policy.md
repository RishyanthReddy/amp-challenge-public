# Assay summary policy

Version 1.0 deliberately provides no aggregated MIC label table. `activity.parquet` and the activity-prediction view preserve every assay observation because species, strain, medium, method, unit, and censoring may differ.

If a downstream task requires a summary, group only observations with the same canonical sequence, canonical species, canonical strain, assay type, normalized unit, and censoring class. Use the median only for at least two exact values after valid µM conversion. Do not mix species, strains, censored bounds, ranges, or unconverted units. Write the aggregation as a separate derived view with its own version and provenance.
