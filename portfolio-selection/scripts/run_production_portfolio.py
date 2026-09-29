"""
Nested Portfolio Selection & Exploratory Model-Score Sampling (Role 07)
Executes:
1. Nested Top 50 (ranks 1-50) and Top 100 (ranks 1-100) selection via Constrained Submodular DPP
2. Deterministic reserves generation (ranks 101-300: Tier 1 niche-matched, Tier 2 contingency)
3. 10,000 Monte Carlo draws under two exploratory draw-size scenarios
4. Summarizes model-threshold counts and score distributions; it does not estimate assay outcomes
5. Exports Parquets, CSVs, and reports/portfolio_lottery_report.md
"""

import sys
import math
import time
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(__file__).resolve().parents[2]

from portfolio.map_elites import MapElitesArchive
from portfolio.dpp_optimizer import DppPortfolioOptimizer


def select_apex_scored_candidates(df: pd.DataFrame, minimum: int = 100) -> pd.DataFrame:
    """Keep only candidates with finite, positive APEX MIC before ranked selection."""
    if "apex_mean_mic" not in df.columns:
        raise KeyError("Master candidate table must contain apex_mean_mic before ranked selection")
    values = pd.to_numeric(df["apex_mean_mic"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & (values > 0)
    if int(mask.sum()) < minimum:
        raise ValueError(
            f"At least {minimum} APEX-scored candidates are required, found {int(mask.sum())}"
        )
    return df.loc[mask].copy().reset_index(drop=True)

def simulate_lottery(candidates_df: pd.DataFrame, draw_k: int = 25, n_draws: int = 10000, seed: int = 42) -> dict:
    """Simulate random draws using a model-based proxy, not measured assay hits."""
    rng = np.random.default_rng(seed)
    N = len(candidates_df)
    if n_draws <= 0 or not 0 < draw_k <= N:
        raise ValueError(f"Require n_draws > 0 and 0 < draw_k <= portfolio size ({N})")
    required = {"lcb_quality", "pred_amp_probability", "pred_toxicity_risk", "synthesizable"}
    missing = required - set(candidates_df.columns)
    if missing:
        raise KeyError(f"Lottery simulation missing model inputs: {sorted(missing)}")

    # Extract candidate metrics
    q_vals = candidates_df["lcb_quality"].to_numpy(dtype=float)
    p_amp = candidates_df["pred_amp_probability"].to_numpy(dtype=float)
    r_tox = candidates_df["pred_toxicity_risk"].to_numpy(dtype=float)
    synth_ok = candidates_df["synthesizable"].astype(bool).to_numpy()
    if not all(np.isfinite(values).all() for values in (q_vals, p_amp, r_tox)):
        raise ValueError("Lottery simulation model inputs must be finite")
    if ((p_amp < 0) | (p_amp > 1)).any() or ((r_tox < 0) | (r_tox > 1)).any():
        raise ValueError("Predicted AMP probability and toxicity risk must be in [0, 1]")

    # A prediction threshold is only a model-based proxy, not an experimentally
    # observed or validated hit label.
    is_model_predicted_candidate = (p_amp >= 0.70) & synth_ok

    draw_hits = []
    draw_mean_q = []
    draw_mean_amp = []
    draw_mean_tox = []

    for _ in range(n_draws):
        idx = rng.choice(N, size=draw_k, replace=False)
        draw_hits.append(np.sum(is_model_predicted_candidate[idx]))
        draw_mean_q.append(np.mean(q_vals[idx]))
        draw_mean_amp.append(np.mean(p_amp[idx]))
        draw_mean_tox.append(np.mean(r_tox[idx]))

    draw_hits = np.array(draw_hits)
    draw_mean_q = np.array(draw_mean_q)

    # 10th percentile and Conditional Value at Risk (CVaR_10)
    q10 = float(np.percentile(draw_mean_q, 10))
    cvar10 = float(np.mean(draw_mean_q[draw_mean_q <= q10]))

    return {
        "portfolio_size": N,
        "draw_size": draw_k,
        "n_draws": n_draws,
        "prob_at_least_1_model_predicted": round(float(np.mean(draw_hits >= 1)), 4),
        "prob_at_least_5_model_predicted": round(float(np.mean(draw_hits >= 5)), 4),
        "prob_at_least_10_model_predicted": round(float(np.mean(draw_hits >= 10)), 4),
        "mean_model_predicted_candidates_in_draw": round(float(np.mean(draw_hits)), 2),
        "mean_drawn_quality": round(float(np.mean(draw_mean_q)), 4),
        "mean_drawn_amp_prob": round(float(np.mean(draw_mean_amp)), 4),
        "mean_drawn_toxicity": round(float(np.mean(draw_mean_tox)), 4),
        "worst_case_q10_quality": round(q10, 4),
        "tail_risk_cvar10": round(cvar10, 4),
    }

def main():
    submission_entry_manifest = ROOT / "reports/submission_entry_manifest.json"
    if submission_entry_manifest.is_file():
        raise RuntimeError(
            "The conservative local submission portfolio has been promoted. This four-domain "
            "research runner would overwrite its canonical Parquet outputs. Use "
            "run_submission_portfolio.py with a new --run-id for an isolated AR/evolution "
            "challenger, or run this research pipeline in a separate worktree."
        )
    print("=======================================================")
    print(" Running Portfolio Selection & Exploratory Score Sampling")
    print("=======================================================\n")

    master_path = MAIN_ROOT / "outputs/master_scored_candidates.csv"
    evaluation_manifest_path = MAIN_ROOT / "shared-evaluator/reports/master_evaluation_manifest.json"
    source_paths = {
        "autoregressive": MAIN_ROOT / "outputs/ar_candidates.csv",
        "vae_latent": MAIN_ROOT / "outputs/vae_candidates.csv",
        "diffusion": MAIN_ROOT / "outputs/diffusion_candidates.csv",
        "evolution": MAIN_ROOT / "outputs/evolution_candidates.csv",
    }
    if not evaluation_manifest_path.is_file():
        raise FileNotFoundError(f"Current Role 06 evaluation manifest is missing: {evaluation_manifest_path}")
    evaluation_manifest = json.loads(evaluation_manifest_path.read_text())

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if sha256_file(master_path) != evaluation_manifest.get("output_sha256"):
        raise ValueError("Master scored candidates do not match the current Role 06 evaluation manifest")
    actual_input_hashes = {domain: sha256_file(path) for domain, path in source_paths.items()}
    if actual_input_hashes != evaluation_manifest.get("input_sha256"):
        raise ValueError("Role 06 source candidate hashes changed after scoring")
    df_raw = pd.read_csv(master_path)
    print(f"Loaded {len(df_raw)} scored master candidates.")
    source_count = len(df_raw)
    df_raw = select_apex_scored_candidates(df_raw, minimum=0)
    print(f"Candidates with finite positive APEX mean MIC: {len(df_raw)}/{source_count}")
    if len(df_raw) < 100:
        raise ValueError("At least 100 APEX-scored candidates are required for ranked portfolio selection")
    print("Ranked selection is restricted to APEX-scored candidates; unscored rows cannot enter Top 100.")

    # 1. MAP-Elites Archiving
    print("[*] Structuring search space via MAP-Elites Quality-Diversity archive...")
    archive = MapElitesArchive()
    arch_res = archive.build_archive(df_raw)
    df_annotated = arch_res["annotated_df"]
    print(f"✓ Archive coverage: {arch_res['occupied_cells']}/{arch_res['total_cells']} cells occupied ({arch_res['coverage_percentage']}%)")

    # 2. Constrained Submodular DPP Optimization
    print("\n[*] Executing Constrained Submodular DPP Optimization...")
    optimizer = DppPortfolioOptimizer(
        alpha=0.60,
        rbf_sigma=2.0,
        det_weight=0.50,
        category_weight=0.30,
        max_domain_quota=35,
    )
    # Select Top 100 with nested Top 50
    dpp_res = optimizer.optimize_portfolio(df_annotated, top_k=100, nested_k=50)
    df_top100 = dpp_res["top100_df"]
    df_top50 = dpp_res["top50_df"]

    print(f"✓ Top 100 selected. Domain breakdown:")
    for dom, cnt in dpp_res["domain_distribution"].items():
        print(f"  - {dom}: {cnt} candidates")

    # 3. Select Reserves (ranks 101-300)
    print("\n[*] Selecting Reserves (ranks 101-300: 100 Tier-1 niche + 100 Tier-2 contingency)...")
    top_seqs_set = set(df_top100["sequence"])
    remaining = df_annotated[(~df_annotated["sequence"].isin(top_seqs_set)) & (df_annotated["max_reference_similarity"] <= 0.80)].copy()

    # Sort remaining by quality descending
    df_reserves = remaining.sort_values(by="lcb_quality", ascending=False).head(200).copy().reset_index(drop=True)
    df_reserves["portfolio_rank"] = np.arange(101, 101 + len(df_reserves))
    df_reserves["reserve_tier"] = np.where(df_reserves["portfolio_rank"] <= 200, "Tier_1_Niche_Matched", "Tier_2_Contingency")
    print(f"✓ Selected {len(df_reserves)} reserves (ranks 101 to {100 + len(df_reserves)}).")

    # 4. Save Outputs
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    main_out = MAIN_ROOT / "outputs"
    main_out.mkdir(parents=True, exist_ok=True)

    df_top100.to_parquet(out_dir / "portfolio_top100.parquet", index=False)
    df_top100.to_csv(out_dir / "portfolio_top100.csv", index=False)
    df_top100.to_csv(main_out / "portfolio_top100.csv", index=False)

    df_top50.to_parquet(out_dir / "portfolio_top50.parquet", index=False)
    df_top50.to_csv(out_dir / "portfolio_top50.csv", index=False)
    df_top50.to_csv(main_out / "portfolio_top50.csv", index=False)

    df_reserves.to_parquet(out_dir / "portfolio_reserves.parquet", index=False)
    df_reserves.to_csv(out_dir / "portfolio_reserves.csv", index=False)
    df_reserves.to_csv(main_out / "portfolio_reserves.csv", index=False)
    print("✓ Saved Top 100, Top 50, and Reserves Parquets & CSVs.")

    # 5. Exploratory random-sampling simulation (not a wet-lab outcome model)
    print("\n[*] Sampling 25 candidates per draw from frozen model scores (10,000 draws)...")
    res_regime_a = simulate_lottery(df_top50, draw_k=25, n_draws=10000, seed=42)
    res_regime_b = simulate_lottery(df_top100, draw_k=25, n_draws=10000, seed=42)

    df_sim_summary = pd.DataFrame([
        {"regime": "Regime_A_Top50_Draw (50% sampling)", **res_regime_a},
        {"regime": "Regime_B_Top100_Draw (25% sampling)", **res_regime_b},
    ])
    df_sim_summary.to_parquet(out_dir / "lottery_simulation_summary.parquet", index=False)
    df_sim_summary.to_csv(out_dir / "lottery_simulation_summary.csv", index=False)

    print("\n================== LOTTERY SIMULATION RESULTS ==================")
    print(df_sim_summary[["regime", "prob_at_least_1_model_predicted", "prob_at_least_5_model_predicted", "mean_model_predicted_candidates_in_draw", "worst_case_q10_quality", "tail_risk_cvar10"]].to_string(index=False))
    print("================================================================\n")

    # 6. Generate Lottery Report Markdown
    rep_md = f"""# Role 07: Exploratory Model-Score Sampling Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Monte Carlo Iterations:** 10,000 draws per regime
**Evaluation:** Model-based Monte Carlo over frozen scores; not a physical assay simulator

---

## 1. Comparative model-score sampling results under both draw sizes

| Metric | Regime A: Top 50 Draw (50% intensity) | Regime B: Top 100 Draw (25% intensity) | Assessment |
| :--- | :--- | :--- | :--- |
| **Probability of ≥1 model-threshold candidate** | **{res_regime_a['prob_at_least_1_model_predicted']*100:.2f}%** | **{res_regime_b['prob_at_least_1_model_predicted']*100:.2f}%** | Depends on predicted AMP probability ≥0.70 and synthesizability flag |
| **Probability of ≥5 model-threshold candidates** | **{res_regime_a['prob_at_least_5_model_predicted']*100:.2f}%** | **{res_regime_b['prob_at_least_5_model_predicted']*100:.2f}%** | Not an experimental hit rate |
| **Mean model-threshold candidates per draw** | **{res_regime_a['mean_model_predicted_candidates_in_draw']}** | **{res_regime_b['mean_model_predicted_candidates_in_draw']}** | Sampling without replacement |
| **Mean Drawn Candidate Quality** | **{res_regime_a['mean_drawn_quality']}** | **{res_regime_b['mean_drawn_quality']}** | Model-derived score only |
| **Worst-Case 10th Percentile ($Q_{{10}}$)** | **{res_regime_a['worst_case_q10_quality']}** | **{res_regime_b['worst_case_q10_quality']}** | Bounded downside tail risk |
| **Conditional Value at Risk ($CVaR_{{10}}$)** | **{res_regime_a['tail_risk_cvar10']}** | **{res_regime_b['tail_risk_cvar10']}** | Resilient against worst 10% draw lottery |

---

## 2. Risk Mitigation & Nested Hierarchy Architecture

1. The two draw sizes are exploratory scenarios; this simulation does not establish an official competition sampling procedure.
2. The threshold label means predicted AMP probability >= 0.70 and a synthesizability flag. It is not a measured hit rate.
3. DPP cosine redundancy uses amino-acid composition vectors; it does not guarantee low Levenshtein sequence similarity or biological success.
"""

    rep_path = ROOT / "reports/portfolio_lottery_report.md"
    with open(rep_path, "w") as f:
        f.write(rep_md)
    print(f"✓ Saved lottery report to {rep_path}")

    portfolio_manifest = {
        "role": "07_portfolio_selection",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "master_candidate_sha256": sha256_file(master_path),
        "role06_manifest_sha256": sha256_file(evaluation_manifest_path),
        "role06_input_sha256": actual_input_hashes,
        "candidate_rows_with_apex_scores": int(len(df_raw)),
        "top100_rows": int(len(df_top100)),
        "top50_rows": int(len(df_top50)),
        "reserves_rows": int(len(df_reserves)),
        "top100_nested_top50": bool(
            set(df_top50["sequence"]) == set(df_top100.head(len(df_top50))["sequence"])
        ),
        "domain_distribution": {str(key): int(value) for key, value in dpp_res["domain_distribution"].items()},
        "parameters": {
            "alpha": 0.60,
            "rbf_sigma": 2.0,
            "det_weight": 0.50,
            "category_weight": 0.30,
            "max_domain_quota": 35,
            "max_pairwise_cosine": 0.92,
        },
        "selection_source_sha256": {
            "run_production_portfolio.py": sha256_file(Path(__file__).resolve()),
            "dpp_optimizer.py": sha256_file(SRC_DIR / "portfolio/dpp_optimizer.py"),
            "map_elites.py": sha256_file(SRC_DIR / "portfolio/map_elites.py"),
        },
        "output_sha256": {
            "portfolio_top100.parquet": sha256_file(out_dir / "portfolio_top100.parquet"),
            "portfolio_top100.csv": sha256_file(out_dir / "portfolio_top100.csv"),
            "portfolio_top50.parquet": sha256_file(out_dir / "portfolio_top50.parquet"),
            "portfolio_reserves.parquet": sha256_file(out_dir / "portfolio_reserves.parquet"),
        },
    }
    portfolio_manifest_path = ROOT / "reports/portfolio_run_manifest.json"
    portfolio_manifest_path.write_text(json.dumps(portfolio_manifest, indent=2) + "\n")
    print(f"✓ Saved portfolio provenance manifest to {portfolio_manifest_path}")

if __name__ == "__main__":
    main()
