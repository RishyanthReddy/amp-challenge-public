# Public release scope

This is a fresh filesystem staging copy, without private Git history. Code files are unchanged from the recorded source commit. Current documentation may reference assets intentionally held outside this candidate and must be reconciled before publication.

## Included
Original source/configuration, existing component license files, documentation, challenge reference and frozen export input/output artifacts. Inclusion is not a license clearance decision.

## Held pending review
Raw and processed third-party training datasets, copied baseline checkpoints, AMP/RBC forest assets and most research outputs. These exclusions mean full training/evaluation is not currently runnable from this candidate. The root frozen-output exporter passed the official validator from a fresh local clone. Two one-sequence CPU inference runs also matched, using the authenticated private checkpoint release; anonymous public access remains unverified.

Eight APEX weights are included and match the organizer starter kit LFS fingerprints.

## Blocking public readiness
- Resolve source-use/distribution questions documented in the review folder.
- Finish exact third-party software/weight license inventory and retained notices.
- Provide public model weights with hash-verified anonymous access; existing private release links do not meet this requirement.
- Complete training disclosure and access/reconstruction instructions for all required inputs; do not claim withheld datasets have been publicly released.
- Review documentation links and perform a complete secret/internal-data scan.
- Verify the final release package against competition requirements. Frozen export is distinct from fresh model inference.
- Owner approved publication of this package on 2026-09-29.

Public repository: https://github.com/RishyanthReddy/amp-challenge-public. No eligibility certification is issued.

## 2026-09-29 disclosure update
The exact AMPlify raw training FASTA is now included under its verified CC BY 4.0 deposit license. AR source identifiers and sequence hashes cover all 26,699 training sequences. Eleven raw-file fingerprints match the retained local snapshots. Other raw datasets are not included; DATA_ACCESS.md describes provider access and missing historic queries/versions. Evaluator membership is now reconstructed with source links and recorded input hashes checked. This does not resolve historical acquisition gaps or other source-use terms.
