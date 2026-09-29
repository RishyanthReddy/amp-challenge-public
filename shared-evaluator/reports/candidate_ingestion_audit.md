# Candidate Ingestion & Reconciliation Audit Report

> Historical record: Earlier four-generator ingestion (2026-09-27). These input counts and hashes precede the later evaluation and the final two-generator submission.
**Date:** 2026-09-27 02:02:11  
**Auditor:** Shared Evaluator & Integration Team  
**Evaluation Target:** Multi-Modal Candidate Reservoir across 4 Completed Generator Roles  

---

## 1. Domain Ingestion Summary

| Domain | File | SHA-256 Checksum | Rows | Unique Seqs | Length Range | Canonical | Exact Ref Matches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Autoregressive** | `ar_candidates.csv` | `92f399b3af935cebeb5e1a9f17739ed16509be061d1a10a77326192ec902cd8a` | 3285 | 3254 | 8-50 | True | 0 |
| **VAE / HydrAMP** | `vae_candidates.csv` | `7d3385647d623d9fcf699a558265e58e4f0eff6166353a25cb9bb80d03fc8cb7` | 3189 | 3027 | 8-25 | True | 0 |
| **Diffusion** | `diffusion_candidates.csv` | `df62a2b596530323d8e277b2f4506657328ad593fd14bd4f17eb3711148ca789` | 3496 | 3496 | 12-37 | True | 0 |
| **Evolution** | `evolution_candidates.csv` | `3c03fdda0ce87619f0b7a45ed79e029ab2f6e4a3617e7a05fb06e307d4cecf28` | 3455 | 3455 | 8-48 | True | 0 |
| **Total Reservoir** | **4 Release Files** | **Reconciled Multi-Modal Pool** | **13425** | **13230** | **8-50 aa** | **100%** | **0** |

---

## 2. Reservoir Status & 50,000 Target Clarification
- **Current Multi-Modal Reservoir:** Exactly **13425 audited candidates** (13230 unique sequences).
- **Exact Reference Overlap:** 0 exact matches across all 39,448 reference sequences in `antibacterial.fasta`.
- **Chemical Invariants:** 100% canonical proteinogenic amino acids; all lengths strictly within $[8, 50]$ residues.
- **Workflow Boundary:** The shared evaluator handles scoring and integrating the multi-modal candidate reservoir, providing unified activity, safety, APEX MIC predictions, and cross-domain overlap analysis. During portfolio selection and assembly, the final library generation generates the remaining sequences to reach 50,000 or packages the submission files.
