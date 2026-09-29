# Role 06: Ground-Truth Evaluator View Audit Report

**Date:** 2026-09-27 02:03:58  
**Source Dataset:** `data-engineering/data/processed/views/evaluator_view.parquet`  
**Data Version:** 1.0 (Frozen by Role 01 Data Engineering)  
**Total Records:** 27509  

---

## 1. Split & Cluster Partitioning

The master dataset partitions sequences to prevent data leakage between sequence families:

| Split | Sequence Count | Percentage | Purpose |
| :--- | :--- | :--- | :--- |
| **core_train_only** | 23458 | 85.3% | Training set for ML classifiers & predictors |
| **train** | 3241 | 11.8% | General training split across diverse clusters |
| **val** | 0 | 0.0% | Hyperparameter tuning & model selection |
| **test** | 405 | 1.5% | Final held-out evaluation & calibration |

---

## 2. Evidence Labels & Missingness Summary

| Label / Dimension | Present Count | Missing / Unknown | Missingness Policy |
| :--- | :--- | :--- | :--- |
| **Antimicrobial Status (`is_amp`)** | 27509 True, 0 False | 0 unknown | Only experimentally confirmed AMPs are labeled positive. |
| **MIC Potency Measurements** | 1214 sequences | 26295 sequences (95.6%) | Missing MIC is explicitly tracked; not imputed as inactive. |
| **Hemolysis / Toxicity Observations** | 630 sequences | 26879 sequences (97.7%) | **Missing toxicity is STRICTLY UNKNOWN, never treated as safe.** |

---

## 3. Evaluator Policy Directives
1. **No Leakage:** Evaluator models must be trained strictly on `core_train_only` and `train` splits. Validation and testing must use strictly held-out `val` and `test` clusters.
2. **Missingness Integrity:** When predicting safety or hemolysis, candidates with missing ground-truth or predictor uncertainty must be flagged as `unknown_safety` rather than awarded favorable scores.
