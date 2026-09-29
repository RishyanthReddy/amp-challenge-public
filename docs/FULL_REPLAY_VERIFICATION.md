# Historical training-to-selection replay

Verified 2026-09-28 in isolated local workspaces and a separate Beam volume. The existing
submission was not promoted, overwritten or regenerated in place.

## Results

| Stage | Directly observed result |
| --- | --- |
| ProGen2 fine-tuning | Full 26,699-sequence training partition, three epochs on RTX 4090; epoch 1 selected by validation loss |
| Checkpoint reproduction | Fresh saved weights SHA-256 exactly matches the submitted checkpoint |
| Historical AR sampling | All 3,500 sequences reproduce in order; every shared output column, including perplexity, identical |
| Candidate audit | All 3,285 retained records reproduce, including their shared audit columns |
| AMP and RBC evaluator training | Both serialized model files reproduce byte-for-byte; AMP ROC-AUC .9762 and RBC OOF ROC-AUC .7280 |
| Evolution | All 3,500 candidates and ancestry CSV reproduce byte-for-byte; 3,493 edges, zero orphan lineages |
| Historical APEX and selection | Rescored candidate pool and reran MAP-Elites/DPP; submitted Top 100 and nested Top 50 reproduce in rank order with no differing shared columns |
| Additional fresh generation | Two completed 50,000-attempt runs from the freshly trained checkpoint; all 100,000 pass PPL <=100 |
| Fresh synthesis screening | 90,040 unique non-reference auxiliary candidates, of which 54,261 pass synthesis |
| Fresh library assembly | 3,965 scored candidates plus 46,035 auxiliary candidates; exactly 50,000 unique sequences, all pass synthesis |
| Fresh final verification | 100 unique top candidates, subset membership, standard alphabet/lengths, clean headers, zero exact matches against 39,448 references; max top/reference .80, max internal .77777778 |
| Repeated fresh selection | Top 100 and Top 50 CSVs identical in two selector runs |
| Original submission | Original FASTA SHA-256 hashes unchanged |

Checkpoint SHA-256: `124b8ea7df5c96cda927bada51c9d26d89f91636d0975fe70e7bc0ef182f9f83`.

Original library: `65183eef34ef86924f311c47de9b9183dabad0634da8586d155c01d81944afba`.
Original top: `7181777317b8a4c7c29841471ae9941dfb0de83b1a0c53c5f0a6bf6d0991e480`.
Fresh test-library: `56bbda22bfb02179477cc6dc482baa565c4427223fa81bcd0f237ed1ebae54b6`.
Fresh test-top: `6e1c9f127d5b4fc4f9c3a2b609b72e493212fb7e7fddf59e2421b9cac5c852d1`.

## Scope and recovery notes

This verifies the chosen private AR/evolution workflow starting from existing curated input
partitions. It does not rerun raw-source ingestion or train the excluded VAE/diffusion
baselines. Their frozen candidate records were used only to preserve historical provenance
exclusion when reproducing the original ranked selection. The separate fresh-library run
has empty baseline inputs and fresh AR/evolution candidates.

The historical auxiliary run attempt schedules were not repeated exactly. Thus the original
Top 100 is reproduced exactly, and full-size fresh library generation is verified, but
byte-identical regeneration of every original library filler is not claimed. Root `generate`
still replays frozen tables; it does not now automatically dispatch GPU training.

The isolated training runner changes the Beam volume, pins runtime packages, and adds hash
metadata. Training mathematics is unchanged. Its local client failed to deserialize a
PyTorch version string subclass after successful remote completion; the completed result
was recovered through the task API. One auxiliary file upload failed before launch and was
retried. Interrupted local clients were recovered from completed remote task results; both
CSV hashes were checked before continuing. No incomplete run was counted as complete.
Perplexity scoring used the available GPU recorded in each manifest (A10G for one batch,
RTX 4090 for the other). No threshold or provenance gate was relaxed.

## Evidence

The private research archive retains `docs/verification/full_replay_20260928/run_manifest.json`
with stage evidence, input hashes, remote task information and exact reproduction checks.
Adjacent records include training summaries, APEX/selection manifests, lineage analysis,
auxiliary run manifests and executed runner snapshots. Those detailed records and raw
replay outputs are not bundled in the public release. This page summarizes the observed
replay; public export and inference checks are documented separately in the execution guide.

This is computational engineering verification, not experimental biological validation.
