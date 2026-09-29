# Prototype Selection & HydrAMP Length Scope Report

**Source Panel:** `data-engineering/data/processed/curated_seed_panel.csv` (100 seeds)
**HydrAMP Hardware Dimension:** `input_shape: [25, 21]` (Max 25 residues)
**Validated Output:** `outputs/prototype_panel_validated.csv`

## Summary of Seed Eligibility
- Total Seeds Audited: 100
- **HydrAMP-Eligible (8 <= L <= 25 residues):** **60 seeds**
- **Incompatible Length (L > 25 residues):** **40 seeds** (flagged and excluded from VAE analogue perturbation to prevent arbitrary truncation)
- **Represented Sequence Clusters:** **2 independent clusters**

## Cluster Distribution of Eligible Seeds
```
cluster_id
not_clustered_reference_overlap_or_non_novelty_safe    59
indel_80_000102                                         1
```

## Quota Balancing Directive
For Mode 2 (Analogue Generation), attempts are distributed uniformly across the **60 eligible seeds** (2 clusters), ensuring balanced coverage without single-family collapse.
