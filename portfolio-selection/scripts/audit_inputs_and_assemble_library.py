"""
Input Audit and 50,000-Sequence Library Assembly (Role 07)
Audits the 13,230 master scored candidates and deterministically assembles
the 50,000-member library.fasta:
- All 13,230 multi-modal candidates (from AR, VAE, Diffusion, Evolution) placed first
- Filled with generative sequences from verified HydrAMP pool to reach exactly 50,000
- 100% unique, 100% canonical amino acids, strictly 8-50 residues, 0 reference overlap
- Exports submission/library.fasta and outputs/full_50k_library.parquet
"""

import sys
import hashlib
import json
import time
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(MAIN_ROOT / "data-engineering/src"))
from amp_data.core import fasta_rows, norm

STANDARD_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    submission_entry_manifest = ROOT / "reports/submission_entry_manifest.json"
    if submission_entry_manifest.is_file():
        raise RuntimeError(
            "The conservative AR/evolution submission artifacts are promoted. This legacy "
            "HydrAMP-fill assembly runner would replace the submission library. Use "
            "scripts/build_submission_artifacts.py with verified ProGen2 pools in a new "
            "run directory, or inspect the current submission_entry_manifest.json."
        )
    print("=======================================================")
    print(" Running Input Audit & 50,000 Library Assembly (Role 07)")
    print("=======================================================\n")

    master_path = MAIN_ROOT / "outputs/master_scored_candidates.csv"
    evaluator_manifest_path = MAIN_ROOT / "shared-evaluator/reports/master_evaluation_manifest.json"
    if not master_path.exists():
        raise FileNotFoundError(f"{master_path} missing!")
    if not evaluator_manifest_path.is_file():
        raise FileNotFoundError(f"Current Role 06 manifest is missing: {evaluator_manifest_path}")
    evaluator_manifest = json.loads(evaluator_manifest_path.read_text())

    candidate_paths = {
        "autoregressive": MAIN_ROOT / "outputs/ar_candidates.csv",
        "vae_latent": MAIN_ROOT / "outputs/vae_candidates.csv",
        "diffusion": MAIN_ROOT / "outputs/diffusion_candidates.csv",
        "evolution": MAIN_ROOT / "outputs/evolution_candidates.csv",
    }
    actual_candidate_hashes = {domain: sha256_file(path) for domain, path in candidate_paths.items()}
    if actual_candidate_hashes != evaluator_manifest.get("input_sha256"):
        raise ValueError("Current role candidate inputs do not match the Role 06 evaluation manifest")

    # 1. Audit Master Candidates
    print("--- 1. Auditing Master Candidate Reservoir ---")
    master_sha = sha256_file(master_path)
    if master_sha != evaluator_manifest.get("output_sha256"):
        raise ValueError("Master scored candidate table does not match its Role 06 manifest")
    df_master = pd.read_csv(master_path)
    print(f"Loaded {len(df_master)} master candidates (SHA-256: {master_sha})")
    if df_master.empty:
        raise ValueError("Master candidate table is empty")
    if len(df_master) != int(evaluator_manifest.get("unique_sequences", -1)):
        raise ValueError("Master candidate row count disagrees with its Role 06 manifest")
    required_master_fields = {
        "sequence", "primary_domain", "contributing_domains", "pred_amp_probability",
        "pred_toxicity_risk", "apex_mean_mic", "max_reference_similarity",
        "passes_novelty_rule_le80", "is_valid", "synthesizable",
    }
    missing_master_fields = required_master_fields - set(df_master.columns)
    if missing_master_fields:
        raise ValueError(f"Master candidate table is missing fields: {sorted(missing_master_fields)}")

    # 2. Check Reference Overlap
    ref_path = MAIN_ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    ref_rows, _, _ = fasta_rows(ref_path)
    ref_seqs = {norm(r["sequence"]) for r in ref_rows}
    if len(ref_seqs) != 39448:
        raise ValueError(f"Expected 39,448 unique reference sequences, found {len(ref_seqs)}")
    print(f"Loaded {len(ref_seqs)} reference sequences from antibacterial.fasta.")

    master_ref_matches = sum(1 for s in df_master["sequence"] if norm(s) in ref_seqs)
    print(f"Master exact reference matches: {master_ref_matches} (100% compliant)")
    if master_ref_matches:
        raise ValueError(f"Master candidate pool contains {master_ref_matches} exact reference matches")

    # 3. Assemble 50,000 Library
    print("\n--- 2. Deterministic 50,000 Library Assembly ---")
    lib_source_path = MAIN_ROOT / "vae-latent-models/outputs/intermediate_hydramp_fastas/library.fasta"
    lib_rows, _, _ = fasta_rows(lib_source_path)
    print(f"Loaded {len(lib_rows)} auxiliary generative sequences from HydrAMP pool.")
    lib_manifest_path = lib_source_path.with_name("library_generation_manifest.json")
    if not lib_manifest_path.is_file():
        raise FileNotFoundError(
            f"HydrAMP auxiliary library provenance is required: {lib_manifest_path}"
        )
    lib_manifest = json.loads(lib_manifest_path.read_text())
    if lib_manifest.get("library_fasta_sha256") != sha256_file(lib_source_path):
        raise ValueError("HydrAMP auxiliary FASTA hash disagrees with its generation manifest")
    if (
        int(lib_manifest.get("requested_count", -1)) != 50_000
        or int(lib_manifest.get("actual_count", -1)) != 50_000
        or int(lib_manifest.get("unique_count", -1)) != 50_000
        or int(lib_manifest.get("exact_reference_overlap_count", -1)) != 0
        or int(lib_manifest.get("reference_count", -1)) != 39_448
    ):
        raise ValueError("HydrAMP auxiliary library manifest does not meet the expected sequence invariants")
    generator_source = MAIN_ROOT / "vae-latent-models/src/hydramp_starter_kit/generate.py"
    runner_source = MAIN_ROOT / "cloud/run_hydramp_library_generation.py"
    if lib_manifest.get("generator_source_sha256") != sha256_file(generator_source):
        raise ValueError("HydrAMP generator source has changed since the auxiliary library was generated")
    if lib_manifest.get("runner_source_sha256") != sha256_file(runner_source):
        raise ValueError("HydrAMP library runner source has changed since the auxiliary library was generated")

    # Build ordered dictionary: all 13,230 multi-modal candidates first, then fill from auxiliary
    library_records = []
    seen = set()

    # Part A: Multi-Modal Candidates
    for idx, row in df_master.iterrows():
        seq = str(row["sequence"]).strip().upper()
        if not (8 <= len(seq) <= 50) or not set(seq) <= STANDARD_AA_SET:
            raise ValueError(f"Invalid master candidate at row {idx}: {seq!r}")
        if seq in seen:
            raise ValueError(f"Master candidate table contains a duplicate sequence at row {idx}")
        if norm(seq) in ref_seqs:
            raise ValueError(f"Master candidate overlaps the official reference at row {idx}")
        seen.add(seq)
        sequence_id = row.get("candidate_id")
        if pd.isna(sequence_id) or str(sequence_id).strip().lower() in {"", "nan", "none"}:
            sequence_id = f"candsha_{hashlib.sha256(seq.encode('ascii')).hexdigest()[:12]}"
        library_records.append({
            "library_index": len(library_records) + 1,
            "sequence_id": str(sequence_id),
            "sequence": seq,
            "length": len(seq),
            "source_tier": "multi_modal_reservoir",
            "primary_domain": row["primary_domain"],
            "contributing_domains": row["contributing_domains"],
            "pred_amp_probability": float(row["pred_amp_probability"]) if pd.notna(row["pred_amp_probability"]) else None,
            "pred_toxicity_risk": float(row["pred_toxicity_risk"]) if pd.notna(row["pred_toxicity_risk"]) else None,
            "apex_mean_mic": float(row["apex_mean_mic"]) if pd.notna(row["apex_mean_mic"]) else None,
            "evaluation_status": "role06_amp_safety_scored_apex_scored" if pd.notna(row["apex_mean_mic"]) else "role06_amp_safety_scored_apex_missing",
        })

    print(f"Added {len(library_records)} multi-modal candidates to library.")

    # Part B: Generative Auxiliary Pool
    fill_count = 0
    for r in lib_rows:
        seq = str(r["sequence"]).strip().upper()
        if seq not in seen and norm(seq) not in ref_seqs and set(seq).issubset(STANDARD_AA_SET) and 8 <= len(seq) <= 50:
            seen.add(seq)
            library_records.append({
                "library_index": len(library_records) + 1,
                "sequence_id": f"lib_gen_{fill_count:05d}",
                "sequence": seq,
                "length": len(seq),
                "source_tier": "generative_auxiliary_hydramp",
                "primary_domain": "vae_latent",
                "contributing_domains": "vae_latent",
                "pred_amp_probability": None,
                "pred_toxicity_risk": None,
                "apex_mean_mic": None,
                "evaluation_status": "unscored_auxiliary_baseline",
            })
            fill_count += 1
            if len(library_records) == 50000:
                break

    print(f"Added {fill_count} generative auxiliary sequences to complete library.")
    if len(library_records) != 50000:
        raise ValueError(f"Expected exactly 50,000 unique library sequences, got {len(library_records)}")

    df_lib = pd.DataFrame(library_records)
    top_path = ROOT / "outputs/portfolio_top100.parquet"
    portfolio_manifest_path = ROOT / "reports/portfolio_run_manifest.json"
    if not portfolio_manifest_path.is_file():
        raise FileNotFoundError(f"Current Role 07 portfolio manifest is missing: {portfolio_manifest_path}")
    portfolio_manifest = json.loads(portfolio_manifest_path.read_text())
    if not top_path.is_file():
        raise FileNotFoundError(f"Required ranked portfolio is missing: {top_path}")
    if portfolio_manifest.get("master_candidate_sha256") != master_sha:
        raise ValueError("Ranked portfolio was not selected from the current Role 06 master table")
    if portfolio_manifest.get("role06_manifest_sha256") != sha256_file(evaluator_manifest_path):
        raise ValueError("Role 07 portfolio manifest does not match the current Role 06 manifest")
    if portfolio_manifest.get("output_sha256", {}).get("portfolio_top100.parquet") != sha256_file(top_path):
        raise ValueError("Top-100 Parquet hash disagrees with the Role 07 portfolio manifest")
    df_top = pd.read_parquet(top_path)
    if len(df_top) != 100 or df_top["sequence"].duplicated().any():
        raise ValueError("Top portfolio must have exactly 100 unique sequences")
    top_set = {str(sequence).strip().upper() for sequence in df_top["sequence"]}
    library_set = set(df_lib["sequence"])
    if not top_set.issubset(library_set):
        raise ValueError(f"Top portfolio has {len(top_set - library_set)} sequences absent from library")
    if len(library_set) != 50000:
        raise ValueError(f"Assembled library has only {len(library_set)} unique sequences")
    if library_set & ref_seqs:
        raise ValueError("Assembled library contains an exact antibacterial reference sequence")

    # 4. Serialize the frozen library table. The repository-root generator is the
    # only source of submission FASTAs.
    print("\n--- 3. Serializing the Library Table ---")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    lib_parquet_path = out_dir / "full_50k_library.parquet"
    df_lib.to_parquet(lib_parquet_path, index=False)
    lib_sha = sha256_file(lib_parquet_path)
    print(f"✓ Saved 50,000-member library table to {lib_parquet_path}")
    print(f"  - {lib_parquet_path}")

    # 5. Generate Audit Report
    report_md = f"""# Role 07: Input Audit & 50,000-Sequence Library Assembly Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Auditor:** Portfolio Selection & Quality-Diversity Team
**Master Candidate Table:** `outputs/master_scored_candidates.csv` (SHA-256: `{master_sha}`)
**HydrAMP Library Source:** `vae-latent-models/outputs/intermediate_hydramp_fastas/library.fasta` (SHA-256: `{lib_manifest['library_fasta_sha256']}`; seed {lib_manifest['base_seed']}; upstream `{lib_manifest['upstream_revision']}`)

**Assembled Library Table:** `outputs/full_50k_library.parquet` (SHA-256: `{lib_sha}`)

---

## 1. Library Composition & Source Tiers

| Source Tier | Sequence Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Multi-Modal Candidate Reservoir** | {len(df_master):,} | {len(df_master)/len(df_lib)*100:.1f}% | Candidate rows from the current master table; APEX coverage is incomplete |
| **Generative Auxiliary Pool** | {fill_count:,} | {fill_count/len(df_lib)*100:.1f}% | HydrAMP-derived baseline sequences, unscored by the shared evaluator |
| **Total Assembled Library** | **{len(df_lib):,}** | **100.0%** | Unique canonical sequences; source composition and scores remain subject to eligibility/re-scoring |

---

## 2. Hard Invariant Verification

| Check | Result | Challenge Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Sequence Count** | {len(df_lib):,} sequences | 50,000 sequences | **PASS** |
| **Uniqueness** | {df_lib['sequence'].nunique():,} unique sequences | 0 duplicates | **PASS** |
| **Alphabet** | 100% canonical proteinogenic amino acids | Standard 20 AAs only | **PASS** |
| **Length Window** | Min = {df_lib['length'].min()}, Max = {df_lib['length'].max()} | $8 \\le L \\le 50$ residues | **PASS** |
| **Reference Overlap** | **0 exact matches** | Exactly 0 across 39,448 reference sequences | **PASS** |
| **Top 100 Subset** | {len(top_set & library_set)}/100 present | Top 100 must be a subset of library | **PASS** |

Auxiliary baseline rows have null AMP, toxicity, and APEX score columns. This table is not
a 50,000-row evaluation result and must not be used to claim those sequences were scored.
"""

    rep_audit = ROOT / "reports/input_and_library_assembly_audit.md"
    with open(rep_audit, "w") as f:
        f.write(report_md)
    print(f"✓ Saved input audit and library assembly report to {rep_audit}")

    assembly_manifest = {
        "role": "07_library_assembly",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_candidate_sha256": master_sha,
        "role06_manifest_sha256": sha256_file(evaluator_manifest_path),
        "role07_portfolio_manifest_sha256": sha256_file(portfolio_manifest_path),
        "hydramp_library_fasta_sha256": lib_manifest["library_fasta_sha256"],
        "hydramp_library_manifest_sha256": sha256_file(lib_manifest_path),
        "library_parquet_sha256": lib_sha,
        "portfolio_top100_parquet_sha256": sha256_file(top_path),
        "library_rows": int(len(df_lib)),
        "unique_library_sequences": int(df_lib["sequence"].nunique()),
        "ranked_top_rows": int(len(df_top)),
        "ranked_top_subset_of_library": bool(top_set.issubset(library_set)),
        "master_candidate_rows": int(len(df_master)),
        "auxiliary_hydramp_rows": int(fill_count),
        "exact_reference_overlap_count": 0,
    }
    assembly_manifest_path = ROOT / "reports/library_assembly_manifest.json"
    assembly_manifest_path.write_text(json.dumps(assembly_manifest, indent=2) + "\n")
    print(f"✓ Saved library assembly provenance manifest to {assembly_manifest_path}")

if __name__ == "__main__":
    main()
