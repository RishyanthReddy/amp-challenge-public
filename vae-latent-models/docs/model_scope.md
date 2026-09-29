# HydrAMP Model Scope & Architectural Interface

**Latent Dimension:** 64 (z in R^64)
**Conditioning Dimension:** 2 (c_amp: binary AMP flag, c_mic: binary low-MIC flag)
**Decoder Total Input Dimension:** 66 ([z, c_amp, c_mic] in R^66)
**Max Sequence Length:** 25 residues
**Vocabulary Size:** 21 (20 standard proteinogenic AAs + padding/stop)
**Auxiliary Classifiers:**
- : Predicts p(AMP) in [0.0, 1.0]
- : Predicts p(low_MIC) in [0.0, 1.0]

## Supported Generation Modes
1. **Mode 1 (Unconstrained):**
   - Sample z ~ N(0, I) in R^64
   - Inverse transform via PCA decomposer: z_pca = decomposer.inverse_transform(z)
   - Append condition [c_amp=1, c_mic=1] -> z_cond in R^66
   - Decode via 
   - Filter via auxiliary classifiers: p(AMP) >= 0.8, p(low_MIC) >= 0.8
2. **Mode 2 (Analogues):**
   - Ingest curated seed peptides from  (<=25 aa)
   - One-hot encode and pad to 25 residues
   - Pass to encoder -> mean z_parent in R^64 and latent uncertainty sigma
   - Perturb: z_prime = z_parent + noise(scale = temp * sigma)
   - Decode with target condition [c_amp=1, c_mic=1]
   - Preserve  and  for every attempt.
