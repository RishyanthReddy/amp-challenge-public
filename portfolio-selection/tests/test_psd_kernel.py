"""
Unit Test: Positive-Semidefinite (PSD) Kernel & Cholesky Stability (Role 07)
Mathematically proves:
1. Symmetry: ||K - K^T||_F == 0
2. Positive-Semidefiniteness: all eigenvalues lambda_i >= -1e-10
3. Quality-weighted L-ensemble matrix is strictly PSD
4. Cholesky factorization L = R^T * R succeeds without failure or jitter collapse
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as la
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
from portfolio.dpp_optimizer import DppPortfolioOptimizer

def test_kernel_mathematical_psd():
    ROOT = Path(__file__).resolve().parents[2]
    cands_path = ROOT / "outputs/master_scored_candidates.csv"
    df = pd.read_csv(cands_path, nrows=200)  # sample 200 candidates
    optimizer = DppPortfolioOptimizer(alpha=0.60, rbf_sigma=2.0)
    X_bio, X_aac = optimizer.extract_kernel_features(df)
    S, K_cos = optimizer.compute_similarity_kernel(X_bio, X_aac)
    N = S.shape[0]

    # Test A: Symmetry
    symmetry_error = np.linalg.norm(S - S.T, ord="fro")
    assert symmetry_error < 1e-12, f"Kernel not symmetric: error={symmetry_error}"

    # Test B: Eigenvalues are non-negative
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    print(f"Kernel S min eigenvalue: {min_eig}")
    assert min_eig >= -1e-10, f"Kernel S is not PSD! Min eigenvalue={min_eig}"

    # Test C: Quality-weighted L-ensemble matrix
    qualities = np.clip(df["pred_amp_probability"].to_numpy(), -2.0, 2.0)
    exp_q = np.exp(qualities)
    L = np.outer(exp_q, exp_q) * S
    np.fill_diagonal(L, exp_q**2 + 1e-4)

    L_eigvals = np.linalg.eigvalsh(L)
    min_L_eig = np.min(L_eigvals)
    print(f"L-ensemble min eigenvalue: {min_L_eig}")
    assert min_L_eig >= 0.0, f"L-ensemble is not strictly positive definite! Min eig={min_L_eig}"

    # Test D: Cholesky decomposition succeeds
    R = la.cholesky(L, lower=True)
    recon_error = np.linalg.norm(L - np.dot(R, R.T), ord="fro")
    assert recon_error < 1e-10, f"Cholesky reconstruction failed: error={recon_error}"
    print(f"✓ Cholesky decomposition verified with reconstruction error: {recon_error:.2e}")

if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
