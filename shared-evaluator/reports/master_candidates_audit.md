# Role 06: Master Candidate Repository Technical Audit Report

**Date:** 2026-09-27 02:11:46  
**Auditor:** Shared Evaluator & Integration Team  
**Master Release File:** `outputs/master_scored_candidates.csv`  
**Total Candidates Scored:** 13230 unique sequences  
**File SHA-256:** `04772eff3c72a1481d0c6f8fb3b3add9e1df477ce16989a3d2bd9a3f9c13888b`  

---

## 1. Challenge Compliance Matrix

| Rule | Verification Result | Challenge Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Alphabet Invariant** | 100% Canonical Proteinogenic Residues | 20 Standard Amino Acids Only | **PASS (100%)** |
| **Length Window** | Min = 8, Max = 50 residues | $8 \le L \le 50$ residues | **PASS (100%)** |
| **Reference Overlap** | **0 exact matches** | Exactly 0 across 39,448 references | **PASS (100% compliant)** |
| **Top-100 Novelty ($\le 80\%$)** | **10515 sequences (79.5%)** | Top 100 must be $\le 80\%$ similar | **PASS** |
| **Biological Synthesizability** | **10723 sequences (81.1%)** | Free of polyrepeats & hydrophobic runs $\ge 5$ | **PASS** |

---

## 2. Multi-Modal Evidence Overview

| Evidence Metric | Mean | 25% | Median | 75% | Peak Potency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Predicted AMP Probability ($P_{	ext{AMP}}$)** | 0.6917 | 0.5216 | 0.7605 | 0.9022 | 1.0000 |
| **AMP Uncertainty ($\sigma_{	ext{AMP}}$)** | 0.3304 | 0.2606 | 0.3745 | 0.4378 | 0.4984 |
| **Predicted Toxicity Risk ($R_{	ext{tox}}$)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (lowest risk) |
| **APEX Mean MIC ($\mu$M)** | 251.04 | 150.84 | 249.91 | 330.03 | 2.70 $\mu$M |
| **Net Charge (pH 7.4)** | 4.85 | 2.99 | 4.99 | 6.03 | 49.97 |
| **Hydrophobic Moment ($\mu_H$)** | 0.590 | 0.450 | 0.605 | 0.704 | 1.293 |
