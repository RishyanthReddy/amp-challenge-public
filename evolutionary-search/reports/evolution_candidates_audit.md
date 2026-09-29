# Role 05: Evolutionary Search Candidates Audit Report

**Date:** 2026-09-27 22:02:05
**Model:** Multi-Family Genetic Algorithm (GA with Family Quotas)
**Total Clean Candidates:** 3500
**Unique Sequences:** 3500 (100.0% unique)

---

## 1. Challenge Compliance Summary

| Check | Result | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Exact Reference Overlap** | **0 matches** | 0 matches across 39,448 antibacterial refs | **PASS (100% compliant)** |
| **Top-100 Novelty ($\le 80\%$)** | **2343 sequences (66.9%)** | Top 100 must be $\le 80\%$ similar | **PASS** |
| **Biological Synthesizability** | **3430 sequences (98.0%)** | Free of polyrepeats, hydrophobic runs $\ge 5$ | **PASS** |
| **Alphabet & Length** | **100% valid (20 canonical AAs, Min=8, Max=49)** | Strict challenge gate ($8 \le L \le 50$) | **PASS** |
| **Ancestry Integrity** | **100% unbroken chains (0 broken links)** | Unbroken DAG back to verified seed | **PASS** |

---

## 2. Biophysical Distributions Across Clean Evolutionary Pool

| Metric | Mean | Min | 25% | Median | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Length (residues)** | 23.0 | 8 | 16.0 | 21.0 | 26.0 | 49 |
| **Composite Fitness** | 0.7340 | -0.4073 | 0.6953 | 0.7792 | 0.8385 | 0.8485 |
| **Net Charge (pH 7.4)** | 5.67 | 0.03 | 4.99 | 5.99 | 6.00 | 8.99 |
| **Hydrophobic Moment ($\mu_H$)** | 0.624 | 0.010 | 0.579 | 0.675 | 0.694 | 1.139 |
| **Boman Index (kcal/mol)** | 1.04 | -1.92 | 0.54 | 1.00 | 1.45 | 5.64 |
| **Ref Similarity (Indel ratio)** | 0.735 | 0.427 | 0.667 | 0.744 | 0.833 | 0.989 |
