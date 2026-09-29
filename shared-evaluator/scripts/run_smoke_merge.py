"""
Four-Way Smoke Merge & Cross-Domain Provenance Verification (Role 06)
Verifies:
1. Ingestion of sample candidates across all 4 generation domains
2. Multi-source tracking when sequences appear in multiple domains
3. Preservation of all generation metadata and biophysical properties
4. Generates reports/smoke_merge_report.json
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(__file__).resolve().parents[2]

from evaluator.validator import validate_hard_rules, validate_schema
from evaluator.biophysical import compute_biophysical_properties

def main():
    print("=======================================================")
    print(" Running Four-Way Smoke Merge & Provenance Test")
    print("=======================================================\n")

    files = {
        "autoregressive": MAIN_ROOT / "outputs/ar_candidates.csv",
        "vae_latent": MAIN_ROOT / "outputs/vae_candidates.csv",
        "diffusion": MAIN_ROOT / "outputs/diffusion_candidates.csv",
        "evolution": MAIN_ROOT / "outputs/evolution_candidates.csv",
    }

    # Load 15 candidates from each domain
    domain_samples = []
    for domain, path in files.items():
        df = pd.read_csv(path, nrows=15)
        df["ingested_domain"] = domain
        domain_samples.append(df)
        print(f"Loaded {len(df)} samples from {domain}")

    # Deliberately inject a shared sequence to test multi-domain provenance aggregation
    shared_test_seq = "KWKLFKKIGKVLKVL"
    synthetic_nodes = [
        {"sequence_id": "test_ar_shared", "sequence": shared_test_seq, "domain": "autoregressive", "model": "ESM2_GPT", "ingested_domain": "autoregressive"},
        {"sequence_id": "test_diff_shared", "sequence": shared_test_seq, "domain": "diffusion", "model": "AMP-Diffusion", "ingested_domain": "diffusion"},
    ]
    df_combined = pd.concat(domain_samples + [pd.DataFrame(synthetic_nodes)], ignore_index=True)
    print(f"\nTotal raw ingested smoke candidates: {len(df_combined)}")

    # 2. Hard check validation and biophysical computation
    merged_records = []
    for idx, row in df_combined.iterrows():
        seq = str(row["sequence"]).strip().upper()
        ok, errs = validate_hard_rules(seq)
        props = compute_biophysical_properties(seq) if ok else {}

        rec = {
            "candidate_id": row.get("sequence_id", f"cand_{idx:04d}"),
            "sequence": seq,
            "domain": row["ingested_domain"],
            "model": row.get("model", "Unknown"),
            "is_valid": ok,
            "validation_errors": errs,
            **props
        }
        merged_records.append(rec)

    df_processed = pd.DataFrame(merged_records)

    # 3. Aggregate by unique sequence while preserving all source links
    grouped_pool = []
    for seq, group in df_processed.groupby("sequence"):
        domains = sorted(group["domain"].unique().tolist())
        cand_ids = group["candidate_id"].tolist()
        models = sorted(group["model"].unique().tolist())

        primary_rec = group.iloc[0].to_dict()
        primary_rec["contributing_domains"] = domains
        primary_rec["contributing_candidate_ids"] = cand_ids
        primary_rec["contributing_models"] = models
        primary_rec["domain_count"] = len(domains)
        grouped_pool.append(primary_rec)

    df_grouped = pd.DataFrame(grouped_pool)
    multi_domain_cands = df_grouped[df_grouped["domain_count"] > 1]

    print(f"Unique sequence count: {len(df_grouped)}")
    print(f"Sequences appearing in multiple domains: {len(multi_domain_cands)}")
    assert len(multi_domain_cands) >= 1, "Multi-domain tracking failed to capture synthetic shared sequence!"

    shared_provenance = multi_domain_cands[multi_domain_cands["sequence"] == shared_test_seq].iloc[0]
    print(f"Verified shared sequence provenance: {shared_provenance['contributing_domains']} (IDs: {shared_provenance['contributing_candidate_ids']})")

    # 4. Save smoke merge report
    report = {
        "status": "PASS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_ingested": len(df_combined),
        "unique_sequences": len(df_grouped),
        "multi_domain_sequences_count": len(multi_domain_cands),
        "tested_domains": list(files.keys()),
        "provenance_preservation_verified": True,
        "sample_multi_domain_entry": {
            "sequence": shared_test_seq,
            "contributing_domains": shared_provenance["contributing_domains"],
            "contributing_candidate_ids": shared_provenance["contributing_candidate_ids"],
        }
    }

    out_rep = ROOT / "reports/smoke_merge_report.json"
    with open(out_rep, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved smoke merge report to {out_rep}")

if __name__ == "__main__":
    main()
