# Shared evaluator

The shared evaluator extracts 27 sequence/biophysical features, scores AMP likelihood and
human erythrocyte hemolysis, keeps a separate hydropathy/moment proxy, and wraps the APEX
pathogen model. These are retrospective screening tools; they do not replace experimental
assays.

## Environment and training

The component environment is defined by `pyproject.toml` and `uv.lock`:

```bash
uv run --project shared-evaluator --locked --group dev \
  pytest -q shared-evaluator/tests
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
The current master table contains 12,756 unique valid candidates. It has complete 11-pathogen
APEX coverage for all 7,655 candidates that pass the novelty and synthesizability gates used
for ranking. The input, code, model, APEX-asset, and output hashes are recorded in
`reports/master_evaluation_manifest.json`; run `uv run --frozen python
cloud/verify_role6_physical.py` from the repository root to verify them. Root `uv run generate`
packages the frozen Parquet artifacts; it does not run this evaluator.
