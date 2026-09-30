# Documentation

## Use the project

- [Model generation](../generation/README.md): hardware, fixed schedules and fresh scoring.
- [Execution guide](running.md): regenerate the submission, inspect its saved artifacts, and run individual pipeline stages.
- [Method](method.md): generation, filtering, evaluation and portfolio selection.
- [Model assets and attribution](assets.md): checkpoint access and third-party components.
- [Submission abstract](FINAL_METHOD_ABSTRACT.md).

## Data and models

- [Data card](../data-engineering/data/DATA_CARD.md) and [source manifest](TRAINING_SOURCE_MANIFEST.csv).
- [Training-source inventory](training_source_scope.json).
- [Activity and hemolysis model cards](../shared-evaluator/reports/model_cards.md).
- [Isolated research replay](RESEARCH_REPLAY.md) and [completion checklist](REPRODUCIBILITY_TODO.md).
- [Source access](DATA_ACCESS.md) and [record membership](training_provenance/README.md).

## Verification evidence

- [Original-library replay and raw-data rebuild](ORIGINAL_LIBRARY_REPLAY.md).
- [Historical training-to-selection replay](FULL_REPLAY_VERIFICATION.md).
- [Checkpoint inference check](INFERENCE_REPLAY.md).
- [Model generation verification](MODEL_GENERATION_VERIFICATION.md): two fresh RTX 4090 runs and exact submitted FASTA hashes.
- [Submitted artifact manifest](FINAL_HANDOFF_MANIFEST.json).
- [Official validator result](official_validator_result.json).
- [Research records](research_records.md): how to interpret older component reports.
- [Public release scope](../PUBLIC_RELEASE_STATUS.md): included assets and additional inputs.

Detailed historical replay records remain in the private research archive.

The evidence records computational results and their scope. It does not establish
experimental peptide activity or safety.
