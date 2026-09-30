# Third-party notices

The root MIT license covers original project software. It does not replace third-party software or data licenses.

- ProGen2 upstream software: BSD-3-Clause, copyright Salesforce.com, Inc. (2022). Retain the full upstream notice in distribution; copied below under licenses/PROGEN_BSD_3_CLAUSE.txt.
- APEX code: retain diffusion-models/apex/LICENSE. Eight bundled weights match the organizer starter kit LFS fingerprints; see docs/license_evidence/APEX_WEIGHT_PROVENANCE.json.
- HydrAMP code: retain vae-latent-models/LICENSE (University of Warsaw, 2022). Baseline checkpoint redistribution is not included in this public release.
- Diffusion component: existing diffusion-models/LICENSE is preserved; the license matches the upstream starter kit byte-for-byte.
- DBAASP: five retained exports are included with acknowledgment under the current source terms; see data-engineering/data/raw/DBAASP/ATTRIBUTION.md.
- UniProt: negative partitions are fetched from a pinned organizer source with CC BY 4.0 attribution; see docs/DATA_ACCESS.md.
- DRAMP: the two retained General files are included under source CC BY 4.0 terms with original references and adjacent attribution. Antibacterial/patent exports are not mirrored.
- dbAMP: its raw file is not mirrored publicly. Pinned retrieval does not resolve the source caveats documented in docs/DATA_ACCESS.md.
- AMPlify: the software license is GPL-3.0; it is not replaced by MIT. The exact training FASTA was separately verified against Zenodo record 7320306, which declares CC BY 4.0. Attribute Chenkai Li, Rene L. Warren and Inanc Birol, cite https://doi.org/10.5281/zenodo.7320306 and https://doi.org/10.1186/s13104-023-06279-1, link https://creativecommons.org/licenses/by/4.0/ and identify project transformations. This finding concerns the dataset, not relicensing the GPL software.

This inventory records checked source notices and declared dependency metadata. It does not certify third-party rights or replace the original license terms.

Registry dependency license metadata is in docs/license_evidence/DEPENDENCY_LICENSE_METADATA.json. Dependencies retain their licenses, including Levenshtein (GPL-2.0-or-later); MIT does not relicense them. No dependency wheels are bundled. This is not an artifact-level legal opinion.
