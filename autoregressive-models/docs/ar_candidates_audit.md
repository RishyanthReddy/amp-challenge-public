# Role 02: Autoregressive Candidates Technical Audit Report

**Date:** 2026-09-27 22:10:11
**Model:** ProGen2-small (151M params, Best Validation Checkpoint — Epoch 1)
**Raw Generated:** 3500
**Exact Reference Matches Quarantined:** 215
**Retained Candidates:** 3285
**Unique Sequences:** 3254 (99.1%)

---

## 1. Challenge Compliance Summary

| Check | Result | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Exact Reference Overlap** | **0 retained; 215 quarantined** | Must be 0 in retained output | **PASS** |
| **Training Set Memorization** | **38 matches** | Low rate indicates genuine de novo generation | **INFO (1.2%)** |
| **Novelty Threshold ($\le 80\%$)** | **2794 sequences (85.1%)** | Top 100 must be $\le 80\%$ | **PASS (Pool available for Top 100)** |
| **Synthesis Feasibility** | **2056 sequences (62.6%)** | Free of polyrepeats, hydrophobic runs $\ge 5$, charge $<1$ | **PASS** |
| **Alphabet & Length** | **100% valid (20 canonical AAs, $8 \le L \le 50$)** | Strict challenge gate | **PASS** |

---

## 2. Biophysical Distributions

| Metric | Mean | Min | 25% | Median | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Length (residues)** | 20.0 | 8 | 13.0 | 19.0 | 21.0 | 50 |
| **Net Charge (pH 7.4)** | 3.24 | -38.98 | 0.89 | 2.78 | 4.61 | 49.97 |
| **Hydrophobic Moment ($\mu_H$)** | 0.505 | 0.012 | 0.345 | 0.483 | 0.640 | 1.268 |
| **Boman Index (kcal/mol)** | 0.81 | -4.92 | -0.17 | 0.74 | 1.69 | 8.95 |
| **Ref Similarity (Indel ratio)** | 0.649 | 0.393 | 0.550 | 0.615 | 0.735 | 0.989 |

---

## 3. Top-Performing Lead Sample Preview

```
       run_id              sequence  length  max_reference_similarity  net_charge_ph7  eisenberg_moment
ar_beam_00000         CLIPPGPFPGRYR      13                  0.636364          1.8876          0.300424
ar_beam_00001       TLAANASVHPLIRSV      15                  0.620690          1.0332          0.489961
ar_beam_00005  PWIHRFINGIRRRWRAIRLW      20                  0.666667          6.0331          0.915583
ar_beam_00008           KFYKLISRRRL      11                  0.640000          4.9912          0.298992
ar_beam_00010   PFLIVYLSLRLTRYILLHR      19                  0.625000          3.0289          0.535140
ar_beam_00011             KSRFIYRRR       9                  0.666667          4.9919          0.167966
ar_beam_00012   KLALIALIPIIPCAIGCIK      19                  0.588235          1.7830          0.351622
ar_beam_00013 GFCWRVCAYRNGKRACYRRCN      21                  0.761905          5.5692          0.719463
ar_beam_00015             YLRKRRWYK       9                  0.705882          4.9890          0.293738
ar_beam_00016        MVPFRWPWWPWRRK      14                  0.769231          3.9941          0.552489
```
