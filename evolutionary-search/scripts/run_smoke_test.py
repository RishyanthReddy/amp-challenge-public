"""
End-to-End Smoke Test for Evolutionary Search (Role 05)
Verifies:
1. Seed loading from evolution_seeds.csv
2. 3 generations of mutation, evaluation, and selection
3. 100% complete parent-child ancestry logging
4. Export of candidates and ancestry
5. Reload verification: unbroken lineage from candidate to seed
6. Saves reports/smoke_test_report.json
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
ROOT = Path(__file__).resolve().parents[1]
from evolution.mutation import mutate
from evolution.evaluator_adapter import EvaluatorAdapter

def run_smoke_test():
    print("=======================================================")
    print(" Running Role 05 End-to-End Evolutionary Smoke Test")
    print("=======================================================\n")

    seeds_path = ROOT / "data/evolution_seeds.csv"
    if not seeds_path.exists():
        raise FileNotFoundError(f"{seeds_path} not found!")

    df_seeds = pd.read_csv(seeds_path)
    # Take 5 seeds across distinct families
    test_seeds = df_seeds.drop_duplicates(subset=["family_id"]).head(5)
    print(f"Selected {len(test_seeds)} test seeds across {test_seeds['family_id'].nunique()} families.")

    evaluator = EvaluatorAdapter(version="1.0.0-smoke")
    rng = np.random.default_rng(42)

    ancestry_edges = []
    candidates_pool = []
    
    # Initialize Population (Generation 0)
    current_pop = []
    for idx, row in test_seeds.iterrows():
        s_id = f"smoke_seed_{row['family_id']:02d}"
        seq = row["sequence"]
        ev = evaluator.evaluate(seq)
        node = {
            "sequence_id": s_id,
            "sequence": seq,
            "domain": "evolution",
            "model": "SmokeGA",
            "generation": 0,
            "parent_id": None,
            "seed_id": s_id,
            "seed_family": int(row["family_id"]),
            "mutation_type": "seed",
            "mutation_pos": -1,
            "mutation_desc": "initial_seed",
            "composite_fitness": ev["composite_fitness"],
            "amp_score": ev["amp_score"],
            "toxicity_risk": ev["toxicity_risk"],
            "net_charge_ph7": ev["net_charge_ph7"],
            "eisenberg_moment": ev["eisenberg_moment"],
            "boman_index": ev["boman_index"],
            "synthesizable": ev["synthesizable"],
        }
        current_pop.append(node)
        candidates_pool.append(node)

    # Run 3 generations of mutation and selection
    cand_counter = 1
    for gen in range(1, 4):
        print(f"--- Generation {gen} (Pop size: {len(current_pop)}) ---")
        new_children = []
        for parent in current_pop:
            # Generate 3 mutant proposals per parent
            for _ in range(3):
                child_seq, op, pos, delta = mutate(parent["sequence"], rng)
                ev = evaluator.evaluate(child_seq)
                c_id = f"smoke_cand_{cand_counter:04d}"
                cand_counter += 1

                child_node = {
                    "sequence_id": c_id,
                    "sequence": child_seq,
                    "domain": "evolution",
                    "model": "SmokeGA",
                    "generation": gen,
                    "parent_id": parent["sequence_id"],
                    "seed_id": parent["seed_id"],
                    "seed_family": parent["seed_family"],
                    "mutation_type": op,
                    "mutation_pos": pos,
                    "mutation_desc": delta,
                    "composite_fitness": ev["composite_fitness"],
                    "amp_score": ev["amp_score"],
                    "toxicity_risk": ev["toxicity_risk"],
                    "net_charge_ph7": ev["net_charge_ph7"],
                    "eisenberg_moment": ev["eisenberg_moment"],
                    "boman_index": ev["boman_index"],
                    "synthesizable": ev["synthesizable"],
                }
                new_children.append(child_node)
                candidates_pool.append(child_node)

                # Record ancestry edge
                ancestry_edges.append({
                    "parent_id": parent["sequence_id"],
                    "child_id": c_id,
                    "seed_id": parent["seed_id"],
                    "generation": gen,
                    "mutation_type": op,
                    "position": pos,
                    "delta": delta,
                    "fitness_delta": round(ev["composite_fitness"] - parent["composite_fitness"], 4),
                })

        # Selection: top 2 per family to ensure family diversity
        df_gen = pd.DataFrame(new_children)
        selected = []
        for fam_id, group in df_gen.groupby("seed_family"):
            top = group.sort_values(by="composite_fitness", ascending=False).head(2)
            selected.extend(top.to_dict(orient="records"))
        current_pop = selected
        print(f"  Selected {len(current_pop)} survivors across {len(set(c['seed_family'] for c in current_pop))} families.")

    # 4. Export artifacts
    df_pool = pd.DataFrame(candidates_pool)
    df_anc = pd.DataFrame(ancestry_edges)

    smoke_cand_file = ROOT / "reports/smoke_candidates.csv"
    smoke_anc_file = ROOT / "reports/smoke_ancestry.csv"
    df_pool.to_csv(smoke_cand_file, index=False)
    df_anc.to_csv(smoke_anc_file, index=False)
    print(f"\n✓ Exported {len(df_pool)} candidates to {smoke_cand_file}")
    print(f"✓ Exported {len(df_anc)} ancestry edges to {smoke_anc_file}")

    # 5. Reload & Ancestry Verification
    reloaded_cands = pd.read_csv(smoke_cand_file)
    reloaded_anc = pd.read_csv(smoke_anc_file)

    parent_map = dict(zip(reloaded_anc["child_id"], reloaded_anc["parent_id"]))
    seed_nodes = set(reloaded_cands[reloaded_cands["generation"] == 0]["sequence_id"])

    unbroken_chains = 0
    broken_chains = 0
    for idx, row in reloaded_cands.iterrows():
        c_id = row["sequence_id"]
        if row["generation"] == 0:
            continue
        # Trace back to seed
        curr = c_id
        path = [curr]
        while curr in parent_map:
            curr = parent_map[curr]
            path.append(curr)
            if curr in seed_nodes:
                break
        if curr in seed_nodes:
            unbroken_chains += 1
        else:
            broken_chains += 1

    print(f"\n--- Lineage Integrity Check ---")
    print(f"Non-seed candidates audited: {unbroken_chains + broken_chains}")
    print(f"Unbroken ancestry chains: {unbroken_chains}")
    print(f"Broken chains: {broken_chains}")
    assert broken_chains == 0, "Broken ancestry chain detected!"

    ledger_stats = evaluator.get_ledger_stats()
    print(f"\nEvaluator Ledger Stats: {ledger_stats}")

    # 6. Save Smoke Test Report JSON
    report = {
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "test_seeds_count": len(test_seeds),
        "generations_completed": 3,
        "total_proposals": len(candidates_pool),
        "total_ancestry_edges": len(ancestry_edges),
        "unbroken_lineage_percentage": 100.0,
        "evaluator_ledger": ledger_stats,
        "output_files": [str(smoke_cand_file), str(smoke_anc_file)],
    }
    report_file = ROOT / "reports/smoke_test_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved smoke test report to {report_file}")

if __name__ == "__main__":
    run_smoke_test()
