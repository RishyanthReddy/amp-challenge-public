# Obtaining the training sources

The original files are retained in the research archive. All 11 raw source files still match the historical source manifest. Missing acquisition dates, provider version labels and export queries remain unknown; file loss is not the issue. Research on 2026-09-29 recovered identical public copies of several inputs.

## Restore and verify inputs

The [download manifest](training_sources.json) records 14 inputs: 11 raw database files and three UniProt negative partitions. It supplements [the historical inventory](TRAINING_SOURCE_MANIFEST.csv) without inventing acquisition dates.

```bash
# Restore all 14 exact inputs: retained snapshots plus pinned public downloads.
uv run python scripts/fetch_training_sources.py

# Verify every input without network access.
uv run python scripts/fetch_training_sources.py --verify-only
```

The first command verifies the included snapshots and fetches missing pinned inputs. The second fails unless all 14 files are present with matching sizes and SHA-256 hashes. Existing mismatches are never overwritten. Downloads are staged and checked before installation; a current export with changed bytes is rejected. Downloaded DRAMP, dbAMP and UniProt files are ignored by Git in this public checkout.

For individual inputs, use `--id dbamp_dbamp_df`, `--id uniprot_train`, or another ID in the manifest. `--root /path/to/isolated-checkout` restores files into a separate workspace. Read the source terms below before use.

## DBAASP: retained exports included

The five original peptide and assay exports are included in `data-engineering/data/raw/DBAASP/`, unchanged and with [attribution](../data-engineering/data/raw/DBAASP/ATTRIBUTION.md). Their original export queries and acquisition dates were not recorded. The [current DBAASP terms](https://dbaasp.dbaasp.niaidprod.net/terms-and-conditions), reviewed 2026-09-29, permit access, copying, adaptation and redistribution with acknowledgment. Cite *DBAASP v3*, DOI [10.1093/nar/gkaa991](https://doi.org/10.1093/nar/gkaa991). These inputs retain their source terms; MIT does not replace them.

## DRAMP: antibacterial copies match, general copies differ

The [DRAMP download service](https://dramp.cpu-bioinfor.org/downloads/) currently supplies antibacterial XLSX/FASTA files that match the project's originals byte-for-byte. The helper pins those URLs and hashes.

Today's general XLSX contains 12,784 rows, compared with 11,687 in the retained file. Of the old IDs, 11,684 sequences match and only 4,824 rows match all 28 shared fields. Both current general files have different hashes. They cannot replace the historical inputs for exact reconstruction. The two unchanged old General files are now included with [source attribution](../data-engineering/data/raw/DRAMP/ATTRIBUTION.md), CC BY 4.0 and original author/reference fields. No exact current-provider download is asserted. File-internal timestamps are not evidence of when the project acquired them.

The [DRAMP home page](https://dramp.cpu-bioinfor.org/) states CC BY 4.0 and separately asks researchers to cite original authors for general/clinical records or obtain authorization for patent AMPs. Retain original record references and this caveat. CC BY 4.0 does not grant patent rights. Matching content or including source IDs does not resolve that caveat. The retained general XLSX has zero accession overlaps with today's 18,715-record Patent export. This supports General/Patent record separation, not patent clearance. The public checkout includes only the retained General files; antibacterial and patent exports are not mirrored.

## dbAMP: identical public archive recovered

The exact `dbamp_df.csv` was found in [BioGenies/CancerGram-analysis](https://github.com/BioGenies/CancerGram-analysis/blob/210206b9762bbc82a45436e9743516753bfaa59f/data/dbamp_df.csv), pinned to commit `210206b9762bbc82a45436e9743516753bfaa59f` (2020-11-05). The downloaded 4,818,154-byte file matches SHA-256 `d7a1eede4c6c9d87605145c6b1854d1f1918564d0db58cb756ee330947f5a688`.

This establishes an exact public retrieval route. It does not establish the project's original acquisition route/date or the provider release that produced it. The archive has no declared root license. The [provider terms](https://ycclab.cuhk.edu.cn/dbAMP/download/LICENSE) state free academic use; a broad redistribution grant has not been established. Use the pinned download under applicable source terms rather than publishing a new mirror.

The retained provenance links 1,725 AR training sequences only to dbAMP. Omitting the source file or its name would not remove those sequences' influence from the existing checkpoint.

## AMPlify: exact CC BY 4.0 deposit included

The authors' [Zenodo deposit 7320306](https://doi.org/10.5281/zenodo.7320306) supplies `AMPlify_AMP_train_common.fa` under CC BY 4.0. The exact file is included with [attribution](../data-engineering/data/raw/amplify/ATTRIBUTION.md); the manifest also provides its verified download URL. Attribute Chenkai Li, Rene L. Warren and Inanc Birol, cite [their paper](https://doi.org/10.1186/s13104-023-06279-1), and indicate preprocessing changes. The dataset license is distinct from AMPlify's GPL software license.

## UniProt-derived evaluator negatives: exact organizer partitions

The three `Uniprot_0_25_{train,val,test}.csv` files are exact matches to the [organizer HydrAMP starter kit](https://github.com/szczurek-lab/hydramp-starter-kit/tree/7804df862872ccc6d09fe01c41bafbca194cfa31/data/training), pinned to commit `7804df862872ccc6d09fe01c41bafbca194cfa31`. The helper restores them under `vae-latent-models/data/training/`, their historical evaluator input path. They are unlabeled proteins used as negative examples, not experimentally confirmed inactive peptides.

A previous guide linked Zenodo 7420189; that record is a HydrAMP software archive, not the verified location of these CSVs. The precise CSV hashes now appear in the download manifest. Retain [UniProt attribution and license](https://www.uniprot.org/help/license) and acknowledge the HydrAMP authors. A new UniProt query is not the same training partition.

## Processed inputs and reconstruction

Data processing is documented in [data-engineering](../data-engineering/README.md). The private archive retains the exact processed AR/evaluator views, toxicity table, sequence table and provenance. All 14 raw/evaluator-negative inputs can now be restored from included snapshots and pinned downloads. Their hashes are disclosed in the training summaries and source-recovery evidence. Most mixed-source processed tables are not mirrored publicly. They can be rebuilt from the restored inputs: [value verification](ORIGINAL_LIBRARY_REPLAY.md) confirms the original columns, dtypes and ordered data, with source-path separators normalized. Both evaluator refits then recover the original model files. DRAMP/dbAMP use questions remain separately unresolved. Changing the data and retraining would create a different checkpoint and submission.

The public release contains the original AMP/RBC forests, so inference with those models does not require retraining. [Model verification](assets.md) and [the replay guide](RESEARCH_REPLAY.md) describe what is executable and what still needs archived inputs. The root exporter is separately reproducible from frozen selected tables.

See [source recovery evidence](verification/source_recovery_20260929/README.md) and [training membership](training_provenance/README.md). Exact hashes and public access improve reproducibility; they do not establish source-use clearance or guarantee co-authorship. The organizers decide eligibility.
