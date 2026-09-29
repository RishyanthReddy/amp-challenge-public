# Role 03: VAE & Latent Models Technical Audit Report

**Date:** 2026-09-27 22:02:04
**Model:** HydrAMP (Epoch 37 Checkpoint, R^64 Latent Space, 2D Conditioning)
**Generation Modes:** Mode 1 (Unconstrained) + Mode 2 (Analogues from 60 Curated Seeds)
**Total Clean Candidates:** 2777
**Unique Sequences:** 2508 (90.3%)

---

## 1. Challenge Compliance Summary

| Check | Result | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Exact Reference Overlap** | **0 matches** | Must be 0 for 50k library | **PASS (100% compliant)** |
| **Novelty Threshold ($\le 80\%$)** | **1379 sequences (49.7%)** | Top 100 must be $\le 80\%$ | **PASS (Available for Top 100)** |
| **Biological Synthesizability** | **1996 sequences (71.9%)** | Free of polyrepeats, hydrophobic runs $\ge 5$, charge $<1$ | **PASS** |
| **Alphabet & Length** | **100% valid (20 canonical AAs, $8 \le L \le 25$)** | Strict challenge gate | **PASS** |
| **Ancestry Tracking** | **100% of Mode 2 analogues have valid `parent_sequence_id`** | Luna mandate | **PASS** |

---

## 2. Mode-by-Mode Performance Comparison

```
                 total_candidates  unique_sequences  mean_length  mean_amp_prob  mean_mic_prob  novel_le80  synthesizable
generation_mode
analogue                     1779              1510    16.973019       0.994477       0.973002         393           1626
unconstrained                 998               998    20.461924       0.993641       0.995930         986            370
```

---

## 3. Biophysical Distributions Across Clean Pool

| Metric | Mean | Min | 25% | Median | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Length (residues)** | 18.2 | 8 | 14.0 | 19.0 | 23.0 | 25 |
| **Net Charge (pH 7.4)** | 5.36 | -4.00 | 3.89 | 5.03 | 6.99 | 17.99 |
| **Hydrophobic Moment ($\mu_H$)** | 0.594 | 0.014 | 0.411 | 0.585 | 0.773 | 1.159 |
| **Boman Index (kcal/mol)** | 1.40 | -2.83 | 0.39 | 1.32 | 2.29 | 8.08 |
| **Ref Similarity (Indel ratio)** | 0.763 | 0.412 | 0.621 | 0.810 | 0.898 | 0.978 |
