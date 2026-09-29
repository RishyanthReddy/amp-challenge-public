"""
Unit Test: Hypergeometric Sampling Mathematics (Role 07)
Validates the Monte Carlo simulation against the exact closed-form Hypergeometric CDF:
  P(X >= 1) = 1 - comb(N - K, 25) / comb(N, 25)
Verifies that empirical estimates agree with the exact distribution within Monte Carlo error.
"""

import sys
import math
from pathlib import Path
import numpy as np
from scipy.stats import hypergeom
import pytest

def test_hypergeometric_lottery_convergence():
    N_pool = 100
    K_hits = 8 # marked candidates in portfolio
    draw_k = 25
    n_sims = 20000

    # 1. Closed-form exact probability from hypergeometric distribution
    # scipy.stats.hypergeom(M, n, N) where M=population, n=marked items, N=draw size.
    exact_prob_ge_1 = 1.0 - hypergeom.pmf(0, N_pool, K_hits, draw_k)
    exact_mean_hits = draw_k * (K_hits / N_pool)
    exact_var_hits = draw_k * (K_hits / N_pool) * (1 - K_hits / N_pool) * ((N_pool - draw_k) / (N_pool - 1))
    exact_std_hits = math.sqrt(exact_var_hits)

    # 2. Monte Carlo simulation
    rng = np.random.default_rng(42)
    pop = np.zeros(N_pool, dtype=int)
    pop[:K_hits] = 1

    sim_hits = []
    for _ in range(n_sims):
        drawn = rng.choice(pop, size=draw_k, replace=False)
        sim_hits.append(np.sum(drawn))

    sim_hits = np.array(sim_hits)
    mc_prob_ge_1 = float(np.mean(sim_hits >= 1))
    mc_mean_hits = float(np.mean(sim_hits))
    mc_std_hits = float(np.std(sim_hits))

    print(f"\n--- Hypergeometric Lottery Verification ---")
    print(f"Exact P(X >= 1): {exact_prob_ge_1:.6f} | Monte Carlo: {mc_prob_ge_1:.6f}")
    print(f"Exact Mean Hits: {exact_mean_hits:.4f} | Monte Carlo: {mc_mean_hits:.4f}")
    print(f"Exact Std Hits:  {exact_std_hits:.4f} | Monte Carlo: {mc_std_hits:.4f}")

    # Assert exact match within Monte Carlo tolerance
    prob_se = math.sqrt(exact_prob_ge_1 * (1 - exact_prob_ge_1) / n_sims)
    mean_se = exact_std_hits / math.sqrt(n_sims)
    variance_se = exact_var_hits * math.sqrt(2 / (n_sims - 1))
    assert abs(mc_prob_ge_1 - exact_prob_ge_1) <= 4 * prob_se + 0.001
    assert abs(mc_mean_hits - exact_mean_hits) <= 4 * mean_se + 0.01
    assert abs(mc_std_hits**2 - exact_var_hits) <= 4 * variance_se + 0.01
    print("✓ Monte Carlo estimates agree with the exact Hypergeometric distribution!")

if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
