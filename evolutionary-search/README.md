# Evolutionary search

This package contains a multi-family genetic search with tournament selection, bounded
mutations, a hand-coded biophysical fitness heuristic, and parent/child ancestry output.
The heuristic is for search guidance; it is not the empirical shared-evaluator activity or hemolysis
model. Final candidates must be rescored by the shared evaluator.

## Inputs and environment

The search requires the curated seed panel and processed data views, which are not bundled
in the public checkout. See [source access](../docs/DATA_ACCESS.md). The command below
is for a research checkout with those inputs restored.

## Run and outputs

The environment is defined by `pyproject.toml` and `uv.lock`. From the repository root:

```bash
uv run --project evolutionary-search --locked \
  python evolutionary-search/scripts/run_production_search.py
```

The search writes candidate sequences, ancestry edges, a generation history, and a run
manifest under `evolutionary-search/` and the root `outputs/` folder. The search has a unique
evaluator-call budget, a maximum generation count, and an early stop after repeated
generations with no new sequences. Its manifest records the termination reason; it may stop
before the requested evaluation budget if mutation stagnates.

`validate_ancestry` checks that all parent and child IDs exist, every child has one edge,
root seeds are generation zero, edge fields match candidate fields, seed lineage stays
within its root, and parent generations strictly precede child generations.

The recorded production run retained 3,500 unique sequences with no exact reference
matches and 3,493 valid child-to-parent edges. Run and lineage summaries are retained under
`reports/`; raw candidate and ancestry tables remain in the private research archive.
The [historical replay](../docs/FULL_REPLAY_VERIFICATION.md) reproduced these outputs.
