# AMP Challenge 2027 data card

> License review update (2026-09-29): the exact AMPlify training FASTA matches the authors' CC BY 4.0 Zenodo deposit 7320306. Older statements below describing its data license as unestablished are superseded by this finding. See the public training disclosure and AMPLIFY_DATA_LICENSE_MATCH.json. Other source questions remain open.

## Scope

This repository provides an auditable data layer for AMP generation and downstream activity/toxicity modelling. It does not train or rank a generative model.

## Sources and provenance

Immutable local inputs are dbAMP (`dbamp_df.csv`), DBAASP peptide and assay exports, DRAMP general/antibacterial FASTA and XLSX exports, and AMPlify's `AMPlify_AMP_train_common.fa`. Every standardized record retains its source database, source file, record identifier, original header/sequence, normalized sequence, and metadata-derived transformation flags in `processed/provenance.csv`.

The official AMP Challenge reference is downloaded from [the challenge repository](https://github.com/szczurek-lab/amp-challenge-2027/blob/main/data/antibacterial.fasta) into `data/challenge/antibacterial.fasta`. It is a novelty-validation constraint, never a declared training source.

APD3 is not present as an explicit local download. It may be indirectly represented by AMPlify overlap, but that cannot be asserted as provenance; obtain a licensed, versioned APD3 export before claiming direct APD3 use.

HydrAMP training records are not merged into this core corpus. The VAE role uses the public HydrAMP starter-kit code and pinned checkpoint/PCA assets documented in `vae-latent-models/README.md`; these are model assets, not source records in this data card. Any separately copied HydrAMP training records need their own versioned provenance and rights review.

## Processing and limitations

Normalization only removes formatting whitespace and uppercases sequences. Non-standard residues, out-of-range lengths, and metadata mentioning chemical modifications are flagged rather than repaired. Exact duplicate canonical sequences retain all provenance. Assay observations are not collapsed: raw units and text remain in activity/toxicity tables.

The split uses deterministic 80%-similarity connected components (seed 42), calculated exhaustively over the eligible 4,051-sequence corpus with the same Indel-ratio semantics as the official validator. Reports also include 70% and 60% components. This protects that declared corpus; a future corpus expansion must rerun clustering rather than reuse the split.

## Licensing and distribution status

The source manifest records file hashes and byte counts for local snapshots, but the exact
download dates and source release versions are unknown for several pre-existing exports.
DBAASP's current terms page, reviewed on 2026-09-27, permits use, adaptation, and redistribution
without restriction with acknowledgement. A separate official PDF at the same provider was
last modified on 2026-09-24 and contains a conflicting non-distribution visitor clause. Obtain
clarification about which terms govern these existing exports and derived datasets before
release.

The current dbAMP download page links a license file last modified on 2026-01-13. It says dbAMP
data are licensed for academic use free of charge, but does not expressly grant public
redistribution or discuss prize-bearing competition use. Confirm those rights for the local
export and derived training assets. The local AMPlify FASTA is byte-identical to the maintainer's
`BirolLab/AMPlify` file at commit `3a07713c25b8a21ef66d31d10e121989d26d9320`. The upstream
repository's GPL-3.0 license covers the program and refers to a per-file `COPYRIGHT` file that
is absent at that commit; a separate license for the training sequence file was not found.
Confirm its data rights instead of assuming the software license applies to the FASTA. DRAMP's
site labels the data CC BY 4.0, but also says patent AMP data requires authorization. The local
`Antibacterial_amps.xlsx` export has 28,702 rows: 16,357 Patent, 5,909 Candidate, 6,164 General,
217 Specific, 49 Clinical, and 6 `Clincial` (as spelled in the source). Its FASTA has the same
28,702 IDs. Do not publish that full export without removing records that require authorization
or obtaining permission. The separate 11,687-row `general_amps.xlsx` export is labelled General;
6,157 of its IDs also occur in the antibacterial workbook and all are classified General there.
The local snapshots have no recorded acquisition date or source release version. Attribute DRAMP
and cite original authors as required. APD3 is not included as a direct local source. Do not
publish the raw corpus or derivatives until the applicable terms are established. See the
source-manifest links and:

- [DBAASP terms webpage](https://dbaasp.dbaasp.niaidprod.net/terms-and-conditions)
- [DBAASP terms PDF](https://dbaasp.dbaasp.niaidprod.net/docs/DBAASP_Terms_And_Conditions.pdf)
- [dbAMP download and license page](https://ycclab.cuhk.edu.cn/dbAMP/download.php)
- [DRAMP data and terms](https://cpu-bioinfor.org/)
- [AMPlify source data at the pinned maintainer commit](https://github.com/BirolLab/AMPlify/tree/3a07713c25b8a21ef66d31d10e121989d26d9320/data)
