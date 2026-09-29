# Diffusion Models Technical Audit Report

**Date:** 2026-09-27 22:02:48
**Model:** AMP-Diffusion (16.54M parameters, ESM-2 8M Embedding Diffusion, 1000 steps)
**Total Clean Candidates:** 3495
**Unique Sequences:** 3495 (100.0%)

---

## 1. Challenge Compliance Summary

| Check | Result | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Exact Reference Overlap** | **0 matches** | Must be 0 for 50k library | **PASS (100% compliant)** |
| **Novelty Threshold ($\le 80\%$)** | **3360 sequences (96.1%)** | Top 100 must be $\le 80\%$ | **PASS (Available for Top 100)** |
| **Biological Synthesizability** | **3101 sequences (88.7%)** | Free of polyrepeats, hydrophobic runs $\ge 5$, charge $<1$ | **PASS** |
| **Alphabet & Length** | **100% valid (20 canonical AAs, Min=12, Max=37)** | Strict challenge gate ($8 \le L \le 50$) | **PASS** |

---

## 2. Biophysical Distributions Across Clean Diffusion Pool

| Metric | Mean | Min | 25% | Median | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Length (residues)** | 24.0 | 12 | 18.0 | 25.0 | 29.0 | 37 |
| **Net Charge (pH 7.4)** | 5.33 | -5.00 | 2.99 | 4.99 | 6.99 | 33.97 |
| **Hydrophobic Moment ($\mu_H$)** | 0.669 | 0.031 | 0.552 | 0.668 | 0.794 | 1.248 |
| **Boman Index (kcal/mol)** | 0.69 | -3.18 | -0.17 | 0.61 | 1.51 | 5.86 |
| **Ref Similarity (Indel ratio)** | 0.609 | 0.426 | 0.550 | 0.594 | 0.645 | 0.968 |
