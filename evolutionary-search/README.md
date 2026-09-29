# Evolutionary search

This package contains a multi-family genetic search with tournament selection, bounded
mutations, a hand-coded biophysical fitness heuristic, and parent/child ancestry output.
The heuristic is for search guidance; it is not the empirical shared-evaluator activity or hemolysis
model. Final candidates must be rescored by the shared evaluator.

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

The current production outputs were generated after these guards were added. The independent
Role 05 physical verifier confirms 3,500 unique retained sequences, zero exact reference
matches, 3,493 valid child-to-parent edges, and an intact ancestry DAG. The current run
manifest, candidate audit, ancestry report, and verifier output are under `reports/`.
