# Role 04: Diffusion Data Requirements & Split Contract Report

**Source View:** `data-engineering/data/processed/views/diffusion_view.parquet`
**Data Version:** 1.0 (frozen by Role 01 Data Engineering)
**Total Sequences in View:** 27,509

---

## 1. Split Allocation & Leakage Verification

| Split Name | Sequence Count | Permitted Usage | Leakage Status |
| :--- | :--- | :--- | :--- |
| `core_train_only` | 23,458 | Primary training corpus & length profiling | Clean |
| `train` | 3,241 | Primary training corpus & length profiling | Clean |
| **Combined Permitted Train** | **26,699** | **Authorized for fitting / length priors** | **LEAKAGE-FREE** |
| `validation` | 405 | Held-out validation / perplexity scoring | Protected (NOT used for sampling) |
| `test` | 405 | Held-out final benchmark evaluation | Protected (NOT used for sampling) |

---

## 2. Empirical Length Distribution in Training Split ($N = 26,699$)

* **Mean Length:** 20.48 residues
* **Standard Deviation:** 9.92 residues
* **Range:** Exactly 8 to 50 residues (100% compliant with challenge limits)
* **Quartiles:**
  * 25th Percentile: 13.0 residues
  * Median (50th): 20.0 residues
  * 75th Percentile: 24.0 residues
* **AMP-Diffusion Operational Regime ($10 \le L \le 40$ residues):**
  * 22,068 sequences (82.7% of the training corpus).
* **Length Sampling Policy:**
  * Sampling requested lengths uniformly from $L_{\text{req}} \in [12, 38]$ ensures that 100% of generated sequences fall strictly within the challenge's $[8, 50]$ residue gate without any padding leakage or truncation artifacts.
