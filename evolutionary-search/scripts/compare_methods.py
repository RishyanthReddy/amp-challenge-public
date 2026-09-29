"""
Matched-Budget Comparison: GA vs Beam Search (Role 05)
Evaluates both algorithms under identical conditions:
- Same 20 seeds across 5 distinct biophysical families
- Exact same unique evaluator call budget (N = 500 each)
- Exact same mutation operators and scoring model
- Compares fitness gain, family diversity, pairwise distance, and synthesizability
- Emits reports/method_comparison.csv
"""

import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
ROOT = Path(__file__).resolve().parents[1]

from evolution.evaluator_adapter import EvaluatorAdapter
from evolution.ga import GeneticAlgorithmSearch
from evolution.beam_search import BeamSearchChallenger

def shannon_entropy(labels: list) -> float:
    if not labels:
        return 0.0
    series = pd.Series(labels)
    probs = series.value_counts(normalize=True)
    return -sum(p * math.log(p) for p in probs)

def mean_pairwise_distance(seqs: list, max_sample: int = 50) -> float:
    if len(seqs) < 2:
        return 0.0
    sample = seqs[:max_sample]
    dists = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dists.append(Levenshtein.distance(sample[i], sample[j]))
    return float(np.mean(dists)) if dists else 0.0

def run_comparison():
    print("=======================================================")
    print(" Running Matched-Budget Search Comparison (Role 05)")
    print(" Baseline: Multi-Family Genetic Algorithm (GA)")
    print(" Challenger: Beam Search with Family Quotas")
    print(" Evaluator Call Budget: 500 unique calls per method")
    print("=======================================================\n")

    seeds_path = ROOT / "data/evolution_seeds.csv"
    df_seeds = pd.read_csv(seeds_path)
    # Select 20 seeds across 5 distinct families (4 per family)
    test_families = df_seeds["family_id"].value_counts().head(5).index.tolist()
    matched_seeds = []
    for f in test_families:
        sub = df_seeds[df_seeds["family_id"] == f].head(4)
        matched_seeds.extend(sub.to_dict(orient="records"))
    print(f"Loaded {len(matched_seeds)} matched seeds across {len(test_families)} families.")

    BUDGET = 500

    # 1. Run GA Baseline
    print("\n[*] Running Genetic Algorithm (GA) baseline...")
    eval_ga = EvaluatorAdapter(version="1.0.0-compare-ga")
    ga_search = GeneticAlgorithmSearch(
        seeds=matched_seeds,
        evaluator=eval_ga,
        budget=BUDGET,
        pop_size_per_family=5,
        offspring_per_parent=3,
        run_id="ga_matched_001",
        seed=42,
    )
    res_ga = ga_search.run()
    ledger_ga = res_ga["ledger"]
    print(f"✓ GA finished {res_ga['generations']} gens | Calls: {ledger_ga['unique_evaluator_calls']} | Candidates: {len(res_ga['candidates'])}")

    # 2. Run Beam Search Challenger
    print("\n[*] Running Beam Search challenger...")
    eval_beam = EvaluatorAdapter(version="1.0.0-compare-beam")
    beam_search = BeamSearchChallenger(
        seeds=matched_seeds,
        evaluator=eval_beam,
        budget=BUDGET,
        beam_width=30,
        expansions_per_parent=4,
        beam_quota_per_family=5,
        run_id="beam_matched_001",
        seed=42,
    )
    res_beam = beam_search.run()
    ledger_beam = res_beam["ledger"]
    print(f"✓ Beam finished {res_beam['rounds']} rounds | Calls: {ledger_beam['unique_evaluator_calls']} | Candidates: {len(res_beam['candidates'])}")

    # 3. Analyze Results
    def analyze_pool(cands: list, name: str) -> dict:
        df = pd.DataFrame(cands)
        non_seeds = df[df["generation"] > 0] if "generation" in df.columns else df[df["round"] > 0]
        seeds_df = df[df.get("generation", df.get("round")) == 0]

        seed_mean_fit = seeds_df["composite_fitness"].mean()
        child_mean_fit = non_seeds["composite_fitness"].mean()
        child_max_fit = non_seeds["composite_fitness"].max()
        fit_gain = child_mean_fit - seed_mean_fit

        # Diversity
        uniq_seqs = df["sequence"].nunique()
        active_fams = df["seed_family"].nunique()
        entropy = shannon_entropy(df["seed_family"].tolist())
        mean_pwd = mean_pairwise_distance(non_seeds.sort_values(by="composite_fitness", ascending=False)["sequence"].tolist())
        synth_rate = non_seeds["synthesizable"].mean()

        return {
            "algorithm": name,
            "evaluator_budget": BUDGET,
            "unique_calls_used": ledger_ga["unique_evaluator_calls"] if name == "GeneticAlgorithm" else ledger_beam["unique_evaluator_calls"],
            "total_candidates_produced": len(df),
            "unique_sequences": uniq_seqs,
            "seed_mean_fitness": round(seed_mean_fit, 4),
            "candidate_mean_fitness": round(child_mean_fit, 4),
            "fitness_gain": round(fit_gain, 4),
            "max_fitness": round(child_max_fit, 4),
            "active_families": active_fams,
            "family_entropy": round(entropy, 3),
            "mean_pairwise_lev_distance": round(mean_pwd, 2),
            "synthesizable_fraction": round(synth_rate, 4),
        }

    ga_stats = analyze_pool(res_ga["candidates"], "GeneticAlgorithm")
    beam_stats = analyze_pool(res_beam["candidates"], "BeamSearch")

    df_comp = pd.DataFrame([ga_stats, beam_stats])
    out_comp = ROOT / "reports/method_comparison.csv"
    df_comp.to_csv(out_comp, index=False)

    print("\n================== SEARCH METHOD COMPARISON ==================")
    print(df_comp.to_string(index=False))
    print("==============================================================")
    print(f"✓ Saved comparison to {out_comp}")

if __name__ == "__main__":
    run_comparison()
