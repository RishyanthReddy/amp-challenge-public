# Training data disclosure

The selected method uses a fine-tuned ProGen2 model and evolutionary search. The existing training-source audit records 26,699 unique training sequences with provenance from DBAASP, DRAMP, dbAMP and AMPlify. Counts overlap across sources and must not be summed. UniProt-derived examples support the activity evaluator; empirical human-erythrocyte observations originate from DBAASP. The challenge antibacterial reference is a validation constraint, not a declared training source. Pretrained model data is distinct from the participant's fine-tuning corpus.

## Public source records
- DBAASP: https://dbaasp.org/ ; current terms https://dbaasp.dbaasp.niaidprod.net/terms-and-conditions ; acknowledge source and cite DOI 10.1093/nar/gkaa991.
- DRAMP: https://dramp.cpu-bioinfor.org/ ; CC BY 4.0 statement with separate patent-data caveat. Clarification remains open.
- dbAMP: https://ycclab.cuhk.edu.cn/dbAMP/download.php ; academic-use statement. Redistribution scope remains open.
- AMPlify: https://github.com/BirolLab/AMPlify ; training FASTA provenance pinned to commit 3a07713c25b8a21ef66d31d10e121989d26d9320 in the existing audit. Exact local training FASTA matched byte-for-byte to the authors' Zenodo deposit 7320306 on 2026-09-29. The deposit declares CC BY 4.0; retain attribution and indicate transformations. Evidence: license_evidence/AMPLIFY_DATA_LICENSE_MATCH.json.
- UniProt: https://www.uniprot.org/help/license ; CC BY 4.0 for copyrightable database content, retaining attribution and applicable third-party rights.
- ProGen2 base: https://huggingface.co/hugohrban/progen2-small ; declared BSD-3-Clause. Mirror revision recorded by the project: 43237a0b733c6629226a079266d2985c9fdce9b7.

## Snapshot limits
Some original snapshot acquisition dates and source release versions were not recorded. Existing source hashes identify local files, but do not establish when or under what terms they were obtained. AR and evaluator membership, source identifiers, sequence fingerprints and partition metadata are included in training_provenance/. Most raw records and processed training tables remain outside the public checkout. Membership metadata does not replace those inputs for retraining.

The inspected records describe publicly accessible sources; a complete inventory is still required before asserting that no non-public source or derived dataset obligations remain. The competition's permissive-release requirement for non-public data is not satisfied merely by linking public database homepages.

## Preprocessing and human decisions
The project's data card describes normalization, duplicate handling, modification flags and deterministic partitions. Computational filtering and family eligibility decisions are described in method.md. Full instructions for obtaining the exact historical inputs remain incomplete in this public release. No additional training or generation was performed for this release preparation.

## Evidence
See training_source_scope.json, existing model cards, TRAINING_SOURCE_MANIFEST.csv and documented reproduction reports. These records retain their original scopes and dates; they are not newly rerun scientific verification.

## Source retrieval guide
See [DATA_ACCESS.md](DATA_ACCESS.md) for provider links, expected files, snapshot hashes and known historical acquisition gaps. The authors do not claim that current database downloads reproduce unknown historical exports.

Current AR training source identifiers and partition metadata are now disclosed in [training_provenance/](training_provenance/README.md). Raw assay records and other raw database snapshots remain excluded. The exact AMPlify file is included under its verified CC BY 4.0 deposit license.
