# Role 04: Diffusion Artifact Audit Report

**Date:** 2026-09-27
**Primary Model:** AMP-Diffusion (Torres et al., *Cell Biomaterials* 2025; Chen et al., *bioRxiv* 2024)
**Repository:** `szczurek-lab/ampdiffusion-starter-kit` (pinned baseline for AMP Challenge 2027)
**License:** MIT License (confirmed in `LICENSE`)

---

## 1. Checkpoint Verification

| Property | Value | Verification Status |
| :--- | :--- | :--- |
| **Artifact Path** | `checkpoint/model.pt` | Verified local presence |
| **Artifact Size** | 132,526,179 bytes (~126.4 MiB) | Matches LFS specification |
| **SHA-256 Hash** | `6a3f347df7c02ff6008ac3d2d4826daeadf7418cb6862a6599b4f710e1d7f8aa` | **100% BIT-FOR-BIT MATCH** |
| **Parameter Count** | 16.54 Million parameters (EMA model) | Verified |

---

## 2. Model Architecture & Latent Diffusion Mechanics

* **Diffusion Domain:** Continuous Gaussian reverse-diffusion in ESM-2 embedding space ($\mathbb{R}^{320}$).
* **Denoising Backbone:** `Denoise_Transformer`
  * Max sequence token positions: `seq_len = 42`
  * Hidden representation dimension: `dim = 320`
* **Diffusion Forward/Reverse Process:** `GaussianDiffusion1D`
  * Trained timesteps ($T_{\text{train}}$): **1,000 steps**
  * Training objective: `pred_v` (velocity parameterization)
  * Sampling schedule: Linear / cosine variance schedule
* **Language Model Decoder:** `fair-esm` `esm2_t6_8M_UR50D`
  * Frozen embedding layer projects sequence tokens to $\mathbb{R}^{320}$.
  * Denoised continuous vectors pass into the pretrained ESM-2 LM classification head.
  * Argmax decoding strictly over the indices of the 20 standard proteinogenic amino acids (`ACDEFGHIKLMNPQRSTVWY`).
* **Sequence Length Control:**
  * Sampled in range 10 to 40 residues (enforcing $8 \le L \le 50$ challenge gate).
  * Trailing positions masked with padding during decoding.

---

## 3. Challenger Route & Ablation Scope

In accordance with Luna's directives:
1. **Primary Route:** AMP-Diffusion default sampling at $T_{\text{infer}} = 1000$ steps.
2. **Inference-Step Ablation Challenger:** Matched comparison at reduced step budget ($T_{\text{infer}} = 250$ / $500$ steps) to quantify the quality-versus-latency curve.
3. **Discrete Sequence Diffusion Fallback:** If continuous embedding diffusion degrades at lower steps, a discrete categorical masked diffusion baseline is available.

---

## 4. Hardware & Runtime Requirements

* **Target Compute:** NVIDIA GeForce RTX 4090 (24GB VRAM) via Beam Cloud.
* **Batch Sampling:** Supports batch size $B=64$ to $B=256$ on GPU.
* **Throughput Target:** High-throughput batch sampling on CUDA.

**Gate Status: GO — Primary artifacts verified, hashes validated, ready for Phase 2.**
