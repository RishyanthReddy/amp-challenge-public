"""
Scaled Production Evolutionary Search (Role 05)
Executes:
1. Multi-Family GA on 50 curated seeds across 9 archetypes
2. Budget-capped execution with generation and stagnation bounds
3. Final lineage validation before export
4. Deduplication while preserving 100% complete parent-child lineage
5. Exports outputs/evolution_candidates.csv, outputs/ancestry.csv, and run manifest
"""

import sys
import json
import time
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(__file__).resolve().parents[2]

from evolution.evaluator_adapter import EvaluatorAdapter
from evolution.ga import GeneticAlgorithmSearch, validate_ancestry
sys.path.append(str(MAIN_ROOT / "data-engineering/src"))
from amp_data.core import fasta_rows, norm
from amp_data.ordering import sort_descending_portable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    print("=======================================================")
    print(" Starting Scaled Production Evolutionary Search (Role 05)")
    print(" Engine: Multi-Family Genetic Algorithm (GA)")
    print(" Target Unique Evaluator Calls: 3,500")
    print(" Checkpointing: Generation-level state persistence")
    print("=======================================================\n")

    seeds_path = ROOT / "data/evolution_seeds.csv"
    df_seeds = pd.read_csv(seeds_path)
    print(f"Loaded {len(df_seeds)} production seeds across {df_seeds['family_id'].nunique()} families.")
    reference_path = MAIN_ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    reference_rows, _, _ = fasta_rows(reference_path)
    reference_set = {norm(row["sequence"]) for row in reference_rows}

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    main_out_dir = MAIN_ROOT / "outputs"
    main_out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    evaluator = EvaluatorAdapter(
        version="1.0.0-prod",
        reference_sequences=reference_set,
    )
    BUDGET = 3500

    ga = GeneticAlgorithmSearch(
        seeds=df_seeds.to_dict(orient="records"),
        evaluator=evaluator,
        budget=BUDGET,
        pop_size_per_family=10,
        offspring_per_parent=3,
        tournament_k=3,
        run_id="evo_prod_001",
        seed=42,
    )

    result = ga.run()
    validate_ancestry(result["candidates"], result["ancestry_edges"])
    elapsed = time.time() - t0
    ledger = result["ledger"]

    print(f"\n✓ Search completed in {elapsed:.1f}s ({result['generations']} generations)")
    print(f"  Total requests: {ledger['total_requests']}")
    print(f"  Unique evaluator calls: {ledger['unique_evaluator_calls']}")
    print(f"  Cache hits: {ledger['cache_hits']} ({ledger['cache_hit_rate']*100:.1f}%)")
    print(f"  Pre-evaluator rejections: {ledger['pre_evaluator_rejections']}")
    print(f"  Raw candidate nodes generated: {len(result['candidates'])}")
    print(f"  Ancestry edges logged: {len(result['ancestry_edges'])}")
    print(f"  Termination reason: {result['termination_reason']}")

    # 1. Deduplication with Ancestry Preservation (Task 11)
    df_raw = pd.DataFrame(result["candidates"])
    # Sort by fitness descending so the best candidate instance is the primary representative
    df_sorted = sort_descending_portable(df_raw, "composite_fitness")
    df_unique = df_sorted.drop_duplicates(subset=["sequence"]).copy()
    exact_ref_mask = df_unique["sequence"].map(lambda sequence: norm(sequence) in reference_set)
    exact_ref_matches_quarantined = int(exact_ref_mask.sum())
    invalid_candidate_mask = ~df_unique["is_valid"].fillna(False).astype(bool)
    invalid_candidate_count = int(invalid_candidate_mask.sum())
    df_unique = df_unique.loc[~exact_ref_mask & ~invalid_candidate_mask].copy()
    df_unique = df_unique.reset_index(drop=True)

    print(f"\n--- Candidate Deduplication ---")
    print(f"Raw candidate records: {len(df_raw)}")
    print(f"Unique sequence candidates retained: {len(df_unique)}")

    # 2. Export Candidate Table (Task 12)
    evo_csv = out_dir / "evolution_candidates.csv"
    evo_main = main_out_dir / "evolution_candidates.csv"
    nodes_csv = out_dir / "evolution_candidate_nodes.csv"
    nodes_main = main_out_dir / "evolution_candidate_nodes.csv"
    df_raw.to_csv(nodes_csv, index=False)
    df_raw.to_csv(nodes_main, index=False)
    df_unique.to_csv(evo_csv, index=False)
    df_unique.to_csv(evo_main, index=False)
    print(f"✓ Saved final candidate pool to:")
    print(f"  - {evo_csv}")
    print(f"  - {evo_main}")

    # 3. Export Complete Ancestry DAG (Task 12)
    df_anc = pd.DataFrame(result["ancestry_edges"])
    anc_csv = out_dir / "ancestry.csv"
    anc_main = main_out_dir / "ancestry.csv"
    df_anc.to_csv(anc_csv, index=False)
    df_anc.to_csv(anc_main, index=False)
    print(f"✓ Saved full lineage DAG ({len(df_anc)} edges) to:")
    print(f"  - {anc_csv}")
    print(f"  - {anc_main}")

    # 4. Save Checkpoint History
    hist_df = pd.DataFrame(result["history"])
    hist_path = ROOT / "reports/generation_history.csv"
    hist_df.to_csv(hist_path, index=False)

    # 5. Production Run Manifest
    manifest = {
        "role": "05_evolutionary_search",
        "algorithm": "Multi-Family Genetic Algorithm (GA)",
        "run_id": "evo_prod_001",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_count": len(df_seeds),
        "families_count": int(df_seeds["family_id"].nunique()),
        "generations_completed": result["generations"],
        "evaluator_call_budget": BUDGET,
        "termination_reason": result["termination_reason"],
        "evaluator_ledger": ledger,
        "raw_candidate_count": len(df_raw),
        "candidate_node_count": len(df_raw),
        "unique_candidate_count": len(df_unique),
        "exact_reference_matches_quarantined": exact_ref_matches_quarantined,
        "invalid_candidates_quarantined": invalid_candidate_count,
        "total_ancestry_edges": len(df_anc),
        "mean_candidate_fitness": round(float(df_unique["composite_fitness"].mean()), 4),
        "max_candidate_fitness": round(float(df_unique["composite_fitness"].max()), 4),
        "synthesizable_percentage": round(float(df_unique["synthesizable"].mean() * 100), 2),
        "output_candidate_file": str(evo_main.relative_to(MAIN_ROOT)),
        "output_candidate_sha256": sha256_file(evo_main),
        "output_candidate_nodes_file": str(nodes_main.relative_to(MAIN_ROOT)),
        "output_candidate_nodes_sha256": sha256_file(nodes_main),
        "output_ancestry_file": str(anc_main.relative_to(MAIN_ROOT)),
        "output_ancestry_sha256": sha256_file(anc_main),
        "seed_file": str(seeds_path.relative_to(MAIN_ROOT)),
        "seed_file_sha256": sha256_file(seeds_path),
    }

    manifest_path = ROOT / "reports/production_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Saved run manifest to {manifest_path}")

if __name__ == "__main__":
    main()
