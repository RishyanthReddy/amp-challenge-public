# Role 05: Method Comparison, Candidate Inspection & Route Decision Report

**Date:** 2026-09-27  
**Auditor:** Bio-AI Engineering Team  
**Evaluated Methods:**
1. Multi-Family Genetic Algorithm (GA) with Strict Per-Family Quotas
2. Beam Search Challenger with Per-Family Diversity Quotas

---

## 1. Equal-Budget Matched Comparison Summary

Both algorithms were executed on the **identical 20 seed peptides** across 5 distinct biophysical archetypes under an exact **500 unique evaluator calls budget**:

| Metric | Multi-Family GA Baseline | Beam Search Challenger | Comparison / Advantage |
| :--- | :--- | :--- | :--- |
| **Evaluator Call Budget** | 500 unique calls | 500 unique calls | Equal (100% matched) |
| **Total Candidates Produced** | 506 candidates | 503 candidates | Comparable volume |
| **Unique Sequences** | 500 unique | 500 unique | 100% unique per method |
| **Starting Seed Mean Fitness** | 0.5601 | 0.5601 | Identical baseline |
| **Candidate Mean Fitness** | **0.7632** | 0.6995 | **GA +0.0637 higher** |
| **Net Fitness Gain ($\Delta$)** | **+0.2031** | +0.1394 | **GA +45.7% higher gain** |
| **Max Fitness Observed** | **0.8478** | 0.8470 | Comparable top peak |
| **Surviving Families** | 5 / 5 (100%) | 5 / 5 (100%) | Both preserve all families |
| **Family Shannon Entropy ($H$)** | 1.608 | 1.609 | Equivalent diversity |
| **Mean Pairwise Distance (aa)** | 17.60 residues | 20.34 residues | High structural divergence |
| **Synthesizable Fraction** | **99.18%** | 98.34% | **GA +0.84% cleaner** |

---

## 2. Score Hacking & Pathological Motif Audit

A manual and automated inspection of top-scoring candidates was conducted to verify that high fitness reflects genuine antimicrobial amphipathicity rather than predictor exploitation:

1. **Poly-Cationic Run Check:** No candidate contains homopolymer runs of Lysine or Arginine $\ge 4$ (e.g. `KKKK` or `RRRR`). The maximum contiguous cationic run observed is 2.
2. **Charge Distribution:** Net charges of top-20 GA candidates remain physiological ($+3.0$ to $+6.5$), avoiding extreme charges ($> +10$) that cause unspecific eukaryotic cytotoxicity.
3. **Hydrophobic / Hydrophilic Alternation:** Top leads display canonical amphipathic periodicity (e.g. `KWKLFKKIGKVLKVL` variants maintaining $[i, i+3, i+4]$ hydrophobic cores with polar basic residues).
4. **Length Distribution:** Mean length is 21.4 residues (range: 12 to 39 residues), well centered within the challenge's $[8, 50]$ window without boundary accumulation.

---

## 3. Production Search Route Decision

- **Selected Production Method:** **Multi-Family Genetic Algorithm (GA)**
- **Decision Rationale:** Under equal computational budgets, the Genetic Algorithm achieved substantially higher average fitness improvement (+0.2031 vs +0.1394) while maintaining strict family diversity (Shannon entropy 1.608 across all 5 families) and superior biological synthesizability (99.18%). Tournament selection with subpopulation quotas prevents beam collapse while allowing robust multi-trajectory exploration.

### Frozen Production Hyperparameters:
- **Seed Pool:** 50 curated seeds across 9 diverse archetypes (`evolution_seeds.csv`).
- **Population Management:** Subpopulation quotas of 10 survivors per family.
- **Offspring Generation:** 3 mutant proposals per parent per generation.
- **Selection Operator:** Tournament selection with $k = 3$.
- **Mutation Distribution:** $P(\text{sub}) = 0.70$, $P(\text{ins}) = 0.15$, $P(\text{del}) = 0.15$ with strict boundary guards at $L=8$ and $L=50$.
- **Stopping Criterion:** Budget of 3,500 unique evaluator calls.
- **Ancestry Tracking:** Mandatory logging of all parent-child directed edges and mutation deltas.
