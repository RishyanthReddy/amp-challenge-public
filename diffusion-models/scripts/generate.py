#!/usr/bin/env python3
"""
Thin Generation Adapter for AMP-Diffusion (Role 04)
Provides a clean interface for sampling sequences from the trained latent diffusion model
with configurable inference steps (1000 DDPM vs 250 DDIM), length range, and output schema.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from ampdiffusion_starter_kit.model import Denoise_Transformer, GaussianDiffusion1D
from ema_pytorch import EMA
import esm

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_SET = set(STANDARD_AA)
PEP_MAX_LEN = 42
EMBED_DIM = 320
DEFAULT_CHECKPOINT = ROOT.parent / "cloud/diffusion_checkpoint/model.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_diffusion_pipeline(checkpoint_path: Path, device: torch.device):
    """Load ESM-2 8M backbone and AMP-Diffusion EMA model with exact signatures."""
    print(f"[+] Loading ESM-2 8M backbone...")
    esm2, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm2 = esm2.to(device).eval()
    std_idxs = [alphabet.get_idx(aa) for aa in STANDARD_AA]

    print(f"[+] Loading AMP-Diffusion weights from {checkpoint_path}...")
    denoise_model = Denoise_Transformer(
        esm_layers=esm2.layers,
        embed_dim=EMBED_DIM,
        pep_max_len=PEP_MAX_LEN,
    )
    diffusion = GaussianDiffusion1D(
        denoise_model,
        seq_length=PEP_MAX_LEN,
        timesteps=1000,
        objective="pred_x0",
        loss_type="l2",
        auto_normalize=False,
        embed_dim=EMBED_DIM,
        self_condition=False,
        device=device,
    ).to(device)

    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ema = EMA(diffusion, beta=0.999, update_every=10)
    ema.to(device)
    ema.load_state_dict(data["ema"])
    ema_model = ema.ema_model.eval()

    return ema_model, esm2, std_idxs

@torch.no_grad()
def decode_embeddings(esm2, sampled_tensor, std_idxs, design_len: int) -> list:
    """Decode continuous ESM-2 embeddings via LM head argmax."""
    seqs = []
    for i in range(sampled_tensor.shape[0]):
        logits = esm2.lm_head(sampled_tensor[i, :])
        aa_logits = logits[:, std_idxs]
        pred = torch.argmax(aa_logits, dim=1)
        s = "".join(STANDARD_AA[j] for j in pred.tolist())[:design_len]
        seqs.append(s)
    return seqs

@torch.no_grad()
def sample_batch(ema_model, esm2, std_idxs, batch_size: int, design_len: int, steps: int = 1000):
    """Sample a batch of sequences with specified step budget."""
    if steps not in (250, 1000):
        raise ValueError("steps must be 250 (DDIM) or 1000 (DDPM)")
    if steps == 1000:
        sampled = ema_model.sample(batch_size=batch_size, design_len=design_len + 2)
    else:
        shape = (batch_size, PEP_MAX_LEN, EMBED_DIM)
        # Configure DDIM timesteps properly
        ema_model.sampling_timesteps = steps
        sampled = ema_model.ddim_sample(shape)
    return decode_embeddings(esm2, sampled, std_idxs, design_len)

def main():
    parser = argparse.ArgumentParser(description="AMP-Diffusion Generation Adapter")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--min-len", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=38)
    parser.add_argument("--max-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs/generated_diffusion.csv"))
    args = parser.parse_args()

    if args.n_sequences <= 0:
        parser.error("--n-sequences must be positive")
    if args.batch_size <= 0 or args.max_rounds <= 0:
        parser.error("--batch-size and --max-rounds must be positive")
    if args.steps not in (250, 1000):
        parser.error("--steps must be 250 (DDIM) or 1000 (DDPM)")
    if not (8 <= args.min_len <= args.max_len <= 50):
        parser.error("length bounds must satisfy 8 <= --min-len <= --max-len <= 50")
    if args.max_len + 2 > PEP_MAX_LEN:
        parser.error(f"--max-len cannot exceed {PEP_MAX_LEN - 2} for this checkpoint")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    ema_model, esm2, std_idxs = load_diffusion_pipeline(checkpoint_path, device)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    records = []
    seen = set()
    t0 = time.time()
    round_idx = 0
    while len(records) < args.n_sequences and round_idx < args.max_rounds:
        cur_b = min(args.batch_size, args.n_sequences - len(records))
        d_len = int(rng.integers(args.min_len, args.max_len + 1))
        seqs = sample_batch(ema_model, esm2, std_idxs, cur_b, d_len, steps=args.steps)
        for s in seqs:
            s_clean = s.strip().upper()
            is_valid = set(s_clean).issubset(STANDARD_AA_SET) and 8 <= len(s_clean) <= 50
            if is_valid and s_clean not in seen:
                seen.add(s_clean)
                records.append({
                    "sequence_id": f"diff_amp_{len(records):05d}",
                    "sequence": s_clean,
                    "domain": "diffusion",
                    "model": "AMP-Diffusion",
                    "checkpoint_sha256": checkpoint_sha256,
                    "requested_length": d_len,
                    "actual_length": len(s_clean),
                    "inference_steps": args.steps,
                    "seed": args.seed,
                    "is_valid": True,
                })
        print(f"Generated {len(records)}/{args.n_sequences} valid sequences (elapsed: {time.time()-t0:.1f}s)")
        round_idx += 1

    if len(records) < args.n_sequences:
        raise RuntimeError(
            f"Generation failed: collected {len(records)}/{args.n_sequences} unique valid sequences "
            f"after {round_idx} rounds (limit {args.max_rounds})."
        )

    df = pd.DataFrame(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✓ Saved {len(df)} valid sequences to {out_path}")

if __name__ == "__main__":
    main()
