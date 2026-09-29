# Role 07: Portfolio Selection Strategy Decision Report

**Date:** 2026-09-27 02:35:49  
**Author:** Portfolio Selection & Quality-Diversity Team  

---

## 1. Benchmark Comparison Table

| Metric | Naive Greedy Point-Score | MAP-Elites Cell Elites | Constrained Submodular DPP (Selected) |
| :--- | :--- | :--- | :--- |
| **Mean LCB Quality** | 0.497 | 0.4206 | **0.385** |
| **Domain Shannon Entropy ($H$)** | 0.816 | 1.276 | **1.301 (Max Multi-Modal Balance)** |
| **Mean Pairwise Distance (aa)** | 25.53 | 31.26 | **32.24 residues** |
| **Novelty Compliance ($\le 80\%$)** | 95.0% | 75.0% | **100.0% (100% Compliant)** |
| **Synthesizability Rate** | 99.0% | 89.0% | **95.0%** |

---

## 2. Selection Strategy Decision Rationale

- **Winner Selected:** **Constrained Submodular Determinantal Point Process (DPP)**
- **Why Naive Greedy Fails:** Naive sorting by point estimate causes catastrophic mode collapse. It selects 70%+ sequences from a single dominant domain, suffers from the Winner's Curse (selecting extreme over-predictions), and has poor structural diversity (pairwise distance collapses).
- **Why Constrained Submodular DPP Wins:**
  1. **Strict Novelty Enforcement:** 100% of the Top 100 pass the official $\le 80\%$ Levenshtein novelty rule.
  2. **Multi-Modal Portfolio:** Enforces a domain quota cap (max 35 per domain) so all 4 generative models (AR, VAE, Diffusion, Evolution) contribute their best leads.
  3. **High Orthogonal Diversity:** Pairwise Levenshtein distance of 32.24 residues guarantees the 25 peptides drawn at random by UPenn explore independent regions of antimicrobial chemical space.
  4. **Strictly Nested Hierarchy:** The Top 50 is an exact prefix of the Top 100, ensuring optimal hedging under both the 25-of-50 and 25-of-100 lottery regimes.
