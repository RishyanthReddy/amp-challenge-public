# Shared evaluator

The shared evaluator extracts 27 sequence/biophysical features, scores AMP likelihood and
human erythrocyte hemolysis, keeps a separate hydropathy/moment proxy, and wraps the APEX
pathogen model. These are retrospective screening tools; they do not replace experimental
assays.

## Inputs and environment

Curated training views, negative partitions and the AMP/RBC forest files are not bundled
in this public checkout. The training and full-scoring commands require these additional
inputs; see [source access](../docs/DATA_ACCESS.md) and
[model cards](reports/model_cards.md). The eight APEX weights are included separately.

## Training workflow

The component environment is defined by `pyproject.toml` and `uv.lock`:

```bash
uv run --project shared-evaluator --locked \
  python shared-evaluator/scripts/train_evaluator_models.py
```

Training saves two Random Forest artifacts and writes `reports/training_data_summary.json`
with input hashes and class/split counts. `reports/model_cards.md` records the target
definitions, held-out metrics, and limitations. The AMP classifier compares curated AMP
sequences with UniProt proteins that are unlabeled negatives; those proteins are not proven
non-AMPs.

The hemolysis model uses only explicit human-erythrocyte `50% Hemolysis` concentration
records. It converts supported mass units with peptide molecular weight, excludes ambiguous
or conflicting measurements, and evaluates grouped five-fold predictions with
>=80%-similarity components kept together. The current run contains 183 unambiguous peptides
and reports out-of-fold ROC-AUC 0.7280. This small, heterogeneous set does not establish
prospective performance. Tree-to-tree spread is not a calibrated confidence interval.

The membrane-disruption proxy remains a separate hand-coded heuristic. Do not merge it into
the empirical hemolysis target or describe it as assay-trained.

## APEX and portfolio integration

`apex_scorer.py` calls the separately locked APEX environment under `diffusion-models/apex/`.
The recorded master table contained 12,756 unique valid candidates with 11-pathogen
APEX coverage for all 7,655 candidates passing the novelty and synthesis ranking gates.
Those totals describe the wider four-generator research pool; the submitted entry retains
only eligible ProGen2/evolution sequences. Input, code, model and output hashes are recorded
in `reports/master_evaluation_manifest.json`. Raw scored tables and forest assets remain in
the private archive, so their physical verifier is not a fresh-public-clone command.
Root `uv run generate` exports the frozen selected tables; it does not run this evaluator.
