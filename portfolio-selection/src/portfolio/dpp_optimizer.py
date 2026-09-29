"""
Constrained Submodular Determinantal Point Process (DPP) Portfolio Optimizer (Role 07)
Selects the Top 100 (and nested Top 50) using:
- Positive-Semidefinite (PSD) composite similarity kernel (RBF biophysical + Linear k-mer)
- Quality-weighted L-ensemble matrix: L_ij = exp(q_i) * S_ij * exp(q_j)
- Greedy submodular log-determinant optimization with category coverage bonuses
- Hard constraints:
  1. Strict novelty gate: max_reference_similarity <= 0.80 (strictly fail-closed)
  2. Pairwise redundancy gate: max cosine similarity <= 0.92 via precomputed Gram matrix (eliminates analogue clustering)
  3. Multi-domain diversity quota: ensures representation across all 4 generative domains
- Nested hierarchy: Ranks 1-50 optimized first; extended to Ranks 51-100 without permutation
"""

from typing import List, Dict, Any, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
REQUIRED_DOMAINS = {"autoregressive", "vae_latent", "diffusion", "evolution"}

class DppPortfolioOptimizer:
    def __init__(
        self,
        alpha: float = 0.60,
        rbf_sigma: float = 2.0,
        det_weight: float = 0.50,
        category_weight: float = 0.30,
        max_domain_quota: int = 35,
        min_domain_quota: int = 1,
        max_pairwise_cosine: float = 0.92,
        max_pairwise_levenshtein: Optional[float] = None,
        required_domains: Optional[Iterable[str]] = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if rbf_sigma <= 0.0:
            raise ValueError(f"rbf_sigma must be positive, got {rbf_sigma}")
        if det_weight < 0.0 or category_weight < 0.0:
            raise ValueError("DPP and category weights must be non-negative")
        if max_domain_quota <= 0:
            raise ValueError("max_domain_quota must be positive")
        if min_domain_quota < 0 or min_domain_quota > max_domain_quota:
            raise ValueError("min_domain_quota must be in [0, max_domain_quota]")
        if not 0.0 <= max_pairwise_cosine <= 1.0:
            raise ValueError("max_pairwise_cosine must be in [0, 1]")
        if max_pairwise_levenshtein is not None and not 0.0 <= max_pairwise_levenshtein <= 1.0:
            raise ValueError("max_pairwise_levenshtein must be in [0, 1]")
        domains = frozenset(REQUIRED_DOMAINS if required_domains is None else required_domains)
        if not domains or any(not isinstance(domain, str) or not domain for domain in domains):
            raise ValueError("required_domains must contain nonempty domain names")

        self.alpha = alpha
        self.rbf_sigma = rbf_sigma
        self.det_weight = det_weight
        self.category_weight = category_weight
        self.max_domain_quota = max_domain_quota
        self.min_domain_quota = min_domain_quota
        self.max_pairwise_cosine = max_pairwise_cosine
        self.max_pairwise_levenshtein = max_pairwise_levenshtein
        self.required_domains = domains

    @staticmethod
    def extract_kernel_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract standardized biophysical features and normalized amino acid frequencies."""
        bio_cols = [
            "net_charge_ph7", "eisenberg_moment", "length",
            "isoelectric_point", "gravy", "boman_index", "instability_index"
        ]
        missing = set(bio_cols + ["sequence"]) - set(df.columns)
        if missing:
            raise KeyError(f"Candidate table missing kernel features: {sorted(missing)}")
        X_bio = df[bio_cols].to_numpy(dtype=np.float64)
        if not np.isfinite(X_bio).all():
            raise ValueError("DPP biophysical kernel features must all be finite")
        means = np.nanmean(X_bio, axis=0)
        stds = np.nanstd(X_bio, axis=0)
        stds[stds == 0] = 1.0
        X_bio_std = np.nan_to_num((X_bio - means) / stds, nan=0.0)

        # Amino acid composition (20-dimensional)
        X_aac = np.zeros((len(df), len(STANDARD_AA)), dtype=np.float64)
        for i, seq in enumerate(df["sequence"]):
            L = max(1, len(seq))
            for j, aa in enumerate(STANDARD_AA):
                X_aac[i, j] = seq.count(aa) / L

        norms = np.linalg.norm(X_aac, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_aac_norm = X_aac / norms

        return X_bio_std, X_aac_norm

    def compute_similarity_kernel(self, X_bio: np.ndarray, X_aac: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute composite PSD similarity kernel: S = alpha * RBF(bio) + (1-alpha) * Cosine(aac)."""
        sq_dist = np.sum(X_bio**2, axis=1, keepdims=True) + np.sum(X_bio**2, axis=1) - 2 * np.dot(X_bio, X_bio.T)
        sq_dist = np.maximum(sq_dist, 0.0)
        K_rbf = np.exp(-sq_dist / (2.0 * (self.rbf_sigma**2)))

        # X_aac rows are normalized frequencies and therefore non-negative; their
        # Gram matrix is already non-negative and PSD. Elementwise clipping of a
        # general Gram matrix would not preserve PSD.
        K_cos = np.dot(X_aac, X_aac.T)
        S = self.alpha * K_rbf + (1.0 - self.alpha) * K_cos
        
        # Enforce exact symmetry and unit diagonal
        S = 0.5 * (S + S.T)
        np.fill_diagonal(S, 1.0)
        return S, K_cos

    def optimize_portfolio(
        self,
        candidates_df: pd.DataFrame,
        top_k: int = 100,
        nested_k: int = 50,
    ) -> Dict[str, Any]:
        """
        Greedy Cholesky DPP selection with category coverage, domain quotas, and vectorized pairwise diversity.
        Returns ordered Top 100, where first 50 entries form the nested Top 50.
        """
        df = candidates_df.copy().reset_index(drop=True)

        if top_k <= 0 or nested_k <= 0 or nested_k > top_k:
            raise ValueError("Require 0 < nested_k <= top_k")
        required = {
            "sequence", "primary_domain", "lcb_quality", "net_charge_ph7",
            "eisenberg_moment", "length", "isoelectric_point", "gravy",
            "boman_index", "instability_index", "pred_amp_probability",
            "pred_toxicity_risk", "max_reference_similarity",
        }
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Candidate table missing required DPP fields: {sorted(missing)}")
        if len(df) < top_k:
            raise ValueError(f"Need at least {top_k} candidates, found {len(df)}")
        if df["primary_domain"].isna().any():
            raise ValueError("Candidate domains must be present")
        df["primary_domain"] = df["primary_domain"].astype(str)
        absent_domains = self.required_domains - set(df["primary_domain"])
        if absent_domains:
            raise ValueError(f"Candidate table is missing required domains: {sorted(absent_domains)}")
        df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
        if df["sequence"].duplicated().any():
            raise ValueError("Candidate table must be unique by normalized sequence")
        aa = set(STANDARD_AA)
        invalid_sequences = [
            seq for seq in df["sequence"]
            if not (8 <= len(seq) <= 50) or not set(seq) <= aa
        ]
        if invalid_sequences:
            raise ValueError(f"Candidate table contains invalid peptide sequences: {invalid_sequences[:3]}")
        df["lcb_quality"] = pd.to_numeric(df["lcb_quality"], errors="coerce")
        if not np.isfinite(df["lcb_quality"].to_numpy(dtype=np.float64)).all():
            raise ValueError("Candidate quality scores must be finite")
        numeric_columns = [
            "net_charge_ph7", "eisenberg_moment", "length", "isoelectric_point",
            "gravy", "boman_index", "instability_index", "pred_amp_probability",
            "pred_toxicity_risk", "max_reference_similarity",
        ]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            values = df[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Candidate feature '{column}' must be finite")
        for column in ("pred_amp_probability", "pred_toxicity_risk"):
            values = df[column].to_numpy(dtype=np.float64)
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Candidate feature '{column}' must be in [0, 1]")
        similarity = df["max_reference_similarity"].to_numpy(dtype=np.float64)
        if ((similarity < 0) | (similarity > 1)).any():
            raise ValueError("max_reference_similarity must be in [0, 1]")

        # Strict fail-closed novelty gate: no production bypass.
        eligible_mask = df["max_reference_similarity"] <= 0.80
        if eligible_mask.sum() < top_k:
            raise ValueError(
                f"Fail-Closed Gate Error: Only {eligible_mask.sum()} candidates pass "
                f"max_reference_similarity <= 0.80, but {top_k} are required!"
            )
        df = df[eligible_mask].copy().reset_index(drop=True)

        available_by_domain = df["primary_domain"].astype(str).value_counts().to_dict()
        quota_capacity = sum(min(count, self.max_domain_quota) for count in available_by_domain.values())
        if quota_capacity < top_k:
            raise ValueError(
                f"Domain quota permits only {quota_capacity} selections, but {top_k} are required"
            )
        if any(available_by_domain.get(domain, 0) < self.min_domain_quota for domain in self.required_domains):
            raise ValueError(
                f"Insufficient candidates to meet minimum domain quota {self.min_domain_quota}: "
                f"{available_by_domain}"
            )
        minimum_required = len(self.required_domains) * self.min_domain_quota
        if minimum_required > top_k:
            raise ValueError("top_k is too small to satisfy required per-domain representation")

        N = len(df)
        print(f"Running DPP optimization over {N} eligible candidates...")
        if self.max_pairwise_levenshtein is not None:
            import Levenshtein
            sequences = df["sequence"].tolist()

        # 2. Extract features and compute quality scores
        X_bio, X_aac = self.extract_kernel_features(df)
        S, K_cos = self.compute_similarity_kernel(X_bio, X_aac)
        
        qualities = df["lcb_quality"].to_numpy(dtype=np.float64)
        exp_q = np.exp(np.clip(qualities, -2.0, 2.0))
        
        L = np.outer(exp_q, exp_q) * S
        np.fill_diagonal(L, exp_q**2 + 1e-4)

        # 3. Category utility vectors for multi-objective coverage
        u_broad = np.clip(df["pred_amp_probability"].to_numpy(dtype=np.float64), 0.0, 1.0)
        u_selectivity = np.clip(1.0 - df["pred_toxicity_risk"].to_numpy(dtype=np.float64), 0.0, 1.0)
        u_cationic = np.clip(df.get("net_charge_ph7", 3.0).to_numpy() / 6.0, 0.0, 1.0)
        u_hydrophobic = np.clip(df.get("eisenberg_moment", 0.4).to_numpy() / 0.6, 0.0, 1.0)

        # 4. Greedy Selection Loop
        selected_indices: List[int] = []
        domain_counts: Dict[str, int] = {}
        c_coverage = np.zeros(4, dtype=np.float64)
        chol_L = np.zeros((top_k, top_k), dtype=np.float64)
        levenshtein_redundancy_mask = np.zeros(N, dtype=bool)

        for step in range(top_k):
            best_gain = -1e9
            best_cand = -1

            # Vectorized pairwise redundancy filter: candidates with cosine similarity > threshold to any selected
            if step > 0:
                max_sim_to_selected = np.max(K_cos[selected_indices, :], axis=0)
                redundancy_mask = max_sim_to_selected > self.max_pairwise_cosine
                redundancy_mask |= levenshtein_redundancy_mask
            else:
                redundancy_mask = np.zeros(N, dtype=bool)

            remaining_slots = top_k - step
            unmet_domains = {
                domain for domain in self.required_domains
                if domain_counts.get(domain, 0) < self.min_domain_quota
            }
            force_required = remaining_slots <= len(unmet_domains)

            for i in range(N):
                if i in selected_indices:
                    continue

                # Constraint 1: Domain Quota
                dom = str(df.loc[i, "primary_domain"])
                if domain_counts.get(dom, 0) >= self.max_domain_quota:
                    continue
                if force_required and dom not in unmet_domains:
                    continue

                # Constraint 2: Vectorized Redundancy Guard
                if redundancy_mask[i]:
                    continue

                # Marginal Gain Computation
                gain_q = qualities[i]
                if step == 0:
                    gain_div = np.log(max(1e-4, L[i, i]))
                else:
                    l_sub = L[selected_indices, i]
                    v = solve_triangular(chol_L[:step, :step], l_sub, lower=True)
                    cond_var = L[i, i] - np.dot(v, v)
                    gain_div = np.log(max(1e-4, cond_var))

                new_cov = c_coverage + np.array([u_broad[i], u_selectivity[i], u_cationic[i], u_hydrophobic[i]])
                gain_cat = np.sum(np.log(1.0 + new_cov) - np.log(1.0 + c_coverage))

                total_gain = gain_q + self.det_weight * gain_div + self.category_weight * gain_cat
                if total_gain > best_gain:
                    best_gain = total_gain
                    best_cand = i

            # Constraints are hard gates. Returning a portfolio that violates them would
            # make the optimizer's output claim misleading, so stop when no feasible item remains.
            if best_cand == -1:
                raise ValueError(
                    f"No candidate satisfies domain and cosine redundancy constraints at selection "
                    f"step {step + 1}/{top_k}; relax parameters or provide a broader candidate pool"
                )

            # Update Cholesky factor
            if step > 0:
                l_sub = L[selected_indices, best_cand]
                v = solve_triangular(chol_L[:step, :step], l_sub, lower=True)
                cond_var_raw = L[best_cand, best_cand] - np.dot(v, v)
                if cond_var_raw < -1e-8:
                    raise np.linalg.LinAlgError(
                        f"DPP conditional variance is materially negative ({cond_var_raw})"
                    )
                cond_var = max(1e-4, cond_var_raw)
                chol_L[step, :step] = v
                chol_L[step, step] = np.sqrt(cond_var)
            else:
                chol_L[0, 0] = np.sqrt(max(1e-4, L[best_cand, best_cand]))

            selected_indices.append(best_cand)
            if self.max_pairwise_levenshtein is not None:
                selected_sequence = sequences[best_cand]
                levenshtein_redundancy_mask |= np.fromiter(
                    (
                        Levenshtein.ratio(sequence, selected_sequence)
                        > self.max_pairwise_levenshtein
                        for sequence in sequences
                    ),
                    dtype=bool,
                    count=N,
                )
            dom = str(df.loc[best_cand, "primary_domain"])
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            c_coverage += np.array([u_broad[best_cand], u_selectivity[best_cand], u_cationic[best_cand], u_hydrophobic[best_cand]])

            if step == nested_k - 1:
                print(f"  ✓ Nested Top {nested_k} selected (Domains: {domain_counts})")

        df_selected = df.iloc[selected_indices].copy().reset_index(drop=True)
        if len(df_selected) != top_k or df_selected["sequence"].duplicated().any():
            raise RuntimeError("DPP did not produce the requested number of unique candidates")
        df_selected["portfolio_rank"] = np.arange(1, top_k + 1)
        df_selected["is_nested_top50"] = df_selected["portfolio_rank"] <= nested_k

        return {
            "top100_df": df_selected,
            "top50_df": df_selected.head(nested_k).copy(),
            "domain_distribution": domain_counts,
            "mean_top100_quality": round(float(df_selected["lcb_quality"].mean()), 4),
            "mean_top50_quality": round(float(df_selected.head(nested_k)["lcb_quality"].mean()), 4),
        }
