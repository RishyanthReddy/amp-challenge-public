# Role 05: Search & Evaluator Interface Contract

**Date:** 2026-09-27  
**Status:** FROZEN  
**Target Domain:** Directed In-Silico Evolutionary Search & Recombination  

---

## 1. Sequence & Chemical Validation Rules

| Rule | Specification | Verification Implementation | Rejection Policy |
| :--- | :--- | :--- | :--- |
| **Alphabet** | Strictly 20 standard proteinogenic amino acids: `ACDEFGHIKLMNPQRSTVWY`. | `set(seq).issubset(STANDARD_AA_SET)` | Reject at generation time (pre-evaluator). |
| **Length Bounds** | Minimum 8 residues, maximum 50 residues ($8 \le L \le 50$). | `8 <= len(seq) <= 50` | Reject at generation time (pre-evaluator). |
| **Exact Reference Overlap** | Zero exact matches to challenge reference library `data-engineering/data/challenge/antibacterial.fasta` (39,448 sequences). | Exact set lookup on normalized sequence string. | Quarantined immediately; never exported to release pool. |
| **Top-100 Novelty Gate** | Levenshtein / Indel similarity $\le 0.80$ against official references. | RapidFuzz `fuzz.ratio` / 100.0 with length-constrained search. | Required for Top-100 ranking; fully documented in candidate pool. |
| **Biological Synthesizability** | Free of synthesis impediments: <br>1. No homopolymer runs $\ge 4$ (e.g. `AAAA`, `LLLL`).<br>2. No hydrophobic runs $\ge 5$ from `{V, I, L, M, F, W, C}`.<br>3. Net charge at pH 7.4 $\ge +1.0$. | Evaluated via `amp_data.synthesis_filter.is_synthesizable`. | Flagged in candidate metadata; prioritizes clean peptides. |

---

## 2. Shared Evaluator Scoring & Ledger Accounting

### A. Scoring Interface
The search queries a multi-component evaluator returning separate biophysical and machine learning estimates:
1. **AMP Heuristic / Probability ($S_{\text{AMP}}$):** Composite antimicrobial likelihood derived from amphipathicity, net charge, and Eisenberg hydrophobic moment.
2. **Biophysical Components:**
   - Net Charge at pH 7.4 ($\ge +1.0$ preferred).
   - Eisenberg Hydrophobic Moment ($\mu_H \ge 0.35$ preferred for alpha-helical membrane insertion).
   - Boman Index ($< 2.50$ kcal/mol preferred for low receptor cross-reactivity).
   - GRAVY (Grand Average of Hydropathy, $-0.8$ to $+0.2$ target range).
3. **Safety / Hemolysis Risk Estimate ($S_{\text{tox}}$):** Penalizes extreme hydrophobicity ($\text{GRAVY} > 0.5$ or $\mu_H > 0.7$).

### B. Evaluator-Call Accounting Ledger
To ensure 100% fair comparison between Genetic Algorithm (GA) baseline and Beam Search challenger:
- **Unique Call:** An evaluation request for a previously unseen sequence. Increments `evaluator_calls_count`.
- **Cache Hit:** An evaluation request for a sequence already scored in the current run or previous generation. Returns cached scores without incrementing the call ledger.
- **Pre-Evaluator Rejection:** Sequences violating alphabet, length, or synthesizability filters are rejected *before* calling the evaluator, conserving budget.
- **Budget Enforcement:** Both GA and Beam Search are stopped when `evaluator_calls_count == BUDGET_LIMIT`.

---

## 3. Provenance & Ancestry Schema

Every candidate produced by Role 05 must link back through an unbroken directed acyclic graph (DAG) to its original starting seed.

### Candidate Schema (`outputs/evolution_candidates.csv`):
- `sequence_id`: Unique identifier (e.g. `evo_amp_00001`).
- `sequence`: Clean uppercase peptide string.
- `domain`: `"evolution"`.
- `model`: Search algorithm (`"GA"` or `"BeamSearch"`).
- `run_id`: Execution batch identifier (e.g. `evo_prod_001`).
- `generation`: Generation number where candidate was created ($G \ge 0$).
- `parent_id`: `sequence_id` of immediate parent ($None$ for seeds).
- `seed_id`: Original prototype seed ID from `prototype_panel.csv`.
- `seed_family`: Cluster ID of the originating seed family (0–11).
- `mutation_type`: Operation that created this sequence (`"seed"`, `"sub"`, `"ins"`, `"del"`).
- `mutation_pos`: Residue index (0-based) where mutation occurred.
- `mutation_desc`: Detailed delta (e.g. `A12K`, `ins_5_R`, `del_8_L`).
- `composite_fitness`: Multi-objective scalar fitness.
- `net_charge_ph7`: Isoelectric net charge.
- `eisenberg_moment`: Amphipathic hydrophobic moment.
- `boman_index`: Protein-binding potential index.
- `gravy`: Grand average of hydropathy.
- `exact_match_reference`: Boolean flag (must be `False`).
- `max_reference_similarity`: Maximum similarity to reference library.
- `passes_novelty_rule_le80`: Boolean flag.
- `synthesizable`: Boolean flag.
- `synthesis_flag`: Reason string if non-synthesizable.

### Lineage Ancestry Schema (`outputs/ancestry.csv`):
- `parent_id`: Parent sequence identifier.
- `child_id`: Child sequence identifier.
- `seed_id`: Originating seed identifier.
- `generation`: Generation index.
- `mutation_type`: Substitution / Insertion / Deletion.
- `position`: Position in parent sequence.
- `old_residue`: Deleted or replaced residue(s).
- `new_residue`: Inserted or replacement residue(s).
- `fitness_delta`: Child fitness minus parent fitness.
