# Obtaining the training sources

## Distribution approach
This candidate discloses third-party public training sources and processing code without mirroring the raw databases, except the exact AMPlify file whose deposit license was verified. The official challenge requires full training-data disclosure and permissive release of any non-public data. We interpret disclosure plus original-source access instructions as an appropriate approach for public data; this interpretation is not an organizer ruling. It does not waive applicable source terms or cover genuinely non-public inputs.

## Access routes

### DBAASP
Visit https://dbaasp.org/ (current service: https://dbaasp.dbaasp.niaidprod.net/). Obtain the peptide CSV/FASTA and associated activity, hemolysis/cytotoxicity and antibiofilm exports corresponding to the filenames below. See the site's search/export interface and API documentation. Exact historical export queries and acquisition dates are not recorded; a current export cannot be assumed identical. Current terms permit redistribution with acknowledgment: https://dbaasp.dbaasp.niaidprod.net/terms-and-conditions . Cite DBAASP v3, DOI 10.1093/nar/gkaa991.

### DRAMP
Use https://dramp.cpu-bioinfor.org/downloads/ . The page provides General and Antibacterial XLSX/FASTA exports. Our historical filenames differ from some current labels: use the manifest to identify the historical expected inputs, not filename matching alone. The site states CC BY 4.0 alongside a patent-AMP authorization caveat; retain this caveat and do not infer patent clearance. We do not redistribute those raw exports in this candidate. Exact historic download versions remain unknown.

### dbAMP
Use the provider's download pages: https://ycclab.cuhk.edu.cn/dbAMP/download2024.php (3.0) and https://ycclab.cuhk.edu.cn/dbAMP/download.php (2.0). Obtain the provider's sequence/annotation export under its terms. Our local dbamp_df.csv is a pre-existing CSV with columns Name, Source, Length, Activity, Hemolytic Activity, UniprotKB ID/AC, Sequence, Description, Taxonomy, Experimental Evidence, Target and PubMed. The original export-to-CSV acquisition procedure and version are not recorded. Do not represent a newly downloaded version as the historical training input. Terms: https://ycclab.cuhk.edu.cn/dbAMP/download/LICENSE . Academic use is stated; the original data remains provider-hosted.

### AMPlify
The authors' deposit https://doi.org/10.5281/zenodo.7320306 contains AMPlify_AMP_train_common.fa under CC BY 4.0. The exact dataset file is accessible from https://zenodo.org/api/records/7320306/files/AMPlify_AMP_train_common.fa/content . Its bytes were checked against our local snapshot on 2026-09-29 and match exactly. SHA-256: a04e28f8d29d1bb4f445a6162e210e0999289c31bf93f2f13c8d2268c8dd9cdc. Attribute Li, Warren and Birol; cite https://doi.org/10.1186/s13104-023-06279-1 and the deposit. Indicate preprocessing changes.

### Evaluator negatives / released HydrAMP data
The existing project uses UniProt-derived negative partitions distributed with HydrAMP, rather than a newly sampled UniProt download. Original asset entry points are https://github.com/szczurek-lab/hydramp and https://doi.org/10.5281/zenodo.7420189 . UniProt terms: https://www.uniprot.org/help/license . The precise partition hashes are recorded in shared-evaluator/reports/training_data_summary.json. Do not substitute a current UniProt query and call it the same training partition. This candidate does not bundle the copied HydrAMP data tree; exact archive-file mapping still needs verification.

## Snapshot inventory
Paths below are relative to data-engineering/. The public source manifest retains unknown dates/versions explicitly.

| Source | Expected local input | SHA-256 |
|---|---|---|
| DBAASP | `data/raw/DBAASP/activity-against-target-species.csv` | `d871e664d194ca155a39d9bbff564b73b30783c8afc53f26a0422c0d0aea24f9` |
| DBAASP | `data/raw/DBAASP/hemolytic-and-cytotoxic-activities.csv` | `987ea27c1ed86ec7ddb6868d94d40d41f26a689cf0b72a9241e6d76968f1e921` |
| DBAASP | `data/raw/DBAASP/peptides-antibiofilm-activities.csv` | `d07f34e5efc1bec91a661ee9df412bc22d4dbd78141b59b2f81b5640b0cbb5a9` |
| DBAASP | `data/raw/DBAASP/peptides-fasta.txt` | `607a3dc32237f0869683532093b10478fba2b391406409235f05212b52ff437d` |
| DBAASP | `data/raw/DBAASP/peptides.csv` | `4377a3434ff0fbb6e52b24d01f32df1b9e74e0b462aa5c0f761f6a51bb518277` |
| DRAMP | `data/raw/DRAMP/Antibacterial_amps.fasta` | `a0d484eb6176e123298d19d06f23ccc6398a1f5c75d169cccbbf7590a6e6c835` |
| DRAMP | `data/raw/DRAMP/Antibacterial_amps.xlsx` | `2b6c79485618439877e06bcaedc79fe4c647fd3be36129cc1cef91a70d0421f5` |
| DRAMP | `data/raw/DRAMP/general_amps.fasta` | `5915e91b3501c41a914a05403dfd2a435af59116ffb738a8423d34995a2b9c26` |
| DRAMP | `data/raw/DRAMP/general_amps.xlsx` | `205b13ba026f54e69d598fa53a54c21a279613d516c70e9cd7598307c8c18925` |
| AMPlify training common export | `data/raw/amplify/AMPlify_AMP_train_common.fa` | `a04e28f8d29d1bb4f445a6162e210e0999289c31bf93f2f13c8d2268c8dd9cdc` |
| dbAMP | `data/raw/dbamp/dbamp_df.csv` | `d7a1eede4c6c9d87605145c6b1854d1f1918564d0db58cb756ee330947f5a688` |

## Processing and reproducibility scope
The existing implementation is documented in data-engineering/README.md and data-engineering/src/amp_data/. It reads immutable inputs and produces normalized records, provenance, assay tables and declared splits. Inspect that documentation and source configuration after obtaining the inputs under their applicable terms. Compare downloaded file hashes with TRAINING_SOURCE_MANIFEST.csv before claiming exact historical reproduction.

The submitted FASTAs and frozen export inputs are distributed separately from training records. Root export repeatability does not depend on downloading raw databases. Exact full historical training reconstruction from current provider downloads is not established because some original snapshots/queries are unknown. No private data should be relabeled public merely because its upstream database is public.

## Included exact snapshot and identifiers
The exact AMPlify training FASTA is included at `data-engineering/data/raw/amplify/AMPlify_AMP_train_common.fa`, with adjacent CC BY 4.0 attribution. Other raw databases are not mirrored. [AR source identifiers](training_provenance/AR_TRAINING_SOURCE_IDS.csv) cover all 26,699 sequences in the current AR training view; [summary](training_provenance/PROVENANCE_SUMMARY.json) records hashes and audit scope.
