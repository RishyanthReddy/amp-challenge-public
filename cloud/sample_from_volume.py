"""
High-Speed Batched Autoregressive Generator on Beam Cloud RTX 4090
Loads the fine-tuned checkpoint directly from Beam Volume (/models/progen2_small_amp_finetuned)
Uses batched KV-cached generation (batch_size=64) to generate 3,000 candidates in ~60-90 seconds.
"""

import os
import sys
import time
import json
import argparse
import hashlib
import math
from pathlib import Path
from typing import Optional
import pandas as pd

from beam import Image, Volume, function

PROGEN2_MODEL_REVISION = "43237a0b733c6629226a079266d2985c9fdce9b7"

# Remote container image
image = Image(
    python_version="python3.10",
    python_packages=[
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "numpy==1.26.4",
        "pandas==2.2.3",
    ],
    commands=[
        "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    ],
)

models_volume = Volume(name="amp-models", mount_path="/models")


@function(
    gpu=["RTX4090", "A100-80", "A10G"],
    image=image,
    memory="24Gi",
    cpu=8,
    volumes=[models_volume],
    timeout=600,
)
def generate_candidates_batched(
    n_generate: int = 3000,
    batch_size: int = 64,
    base_seed: int = 42,
    checkpoint_name: str = "progen2_small_amp_best_val",
    max_perplexity: Optional[float] = None,
) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from tokenizers import Tokenizer
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Generator running on {gpu_name} (PyTorch {torch.__version__}, CUDA {torch.version.cuda})")

    if n_generate <= 0:
        raise ValueError("n_generate must be positive")
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be in [1, 64]")
    if max_perplexity is not None and (not math.isfinite(max_perplexity) or max_perplexity <= 0):
        raise ValueError("max_perplexity must be positive when provided")
    if checkpoint_name not in {"progen2_small_amp_best_val", "progen2_small_amp_finetuned"}:
        raise ValueError(f"Unsupported checkpoint name: {checkpoint_name}")
    torch.use_deterministic_algorithms(True, warn_only=True)

    ckpt_dir = f"/models/{checkpoint_name}"
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_dir}!")

    weights_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Expected checkpoint weights at {weights_path}")
    digest = hashlib.sha256()
    with open(weights_path, "rb") as weights_file:
        for chunk in iter(lambda: weights_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    checkpoint_sha256 = digest.hexdigest()

    print(f"[+] Loading fine-tuned model from {ckpt_dir}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        code_revision=PROGEN2_MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_pretrained(
            "hugohrban/progen2-small", revision=PROGEN2_MODEL_REVISION
        )
    tokenizer.no_padding()

    START_TOKEN = "1"
    END_TOKEN = "2"
    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    MIN_LENGTH = 8
    MAX_LENGTH = 50

    start_id = tokenizer.encode(START_TOKEN).ids[0]
    end_id = tokenizer.encode(END_TOKEN).ids[0]
    vocab = tokenizer.get_vocab()
    standard_aa_ids = torch.tensor([vocab[aa] for aa in STANDARD_AAS if aa in vocab], device=device)

    print(f"[+] Model and Tokenizer loaded successfully. Starting Batched Generation...")

    def score_perplexity_batch(sequences: list[str]) -> list[float]:
        token_rows = [tokenizer.encode(START_TOKEN + seq + END_TOKEN).ids for seq in sequences]
        max_tokens = max(map(len, token_rows))
        input_ids = torch.full(
            (len(token_rows), max_tokens), end_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_idx, tokens in enumerate(token_rows):
            input_ids[row_idx, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            attention_mask[row_idx, :len(tokens)] = 1
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
            targets = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:].to(dtype=logits.dtype)
            losses = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape(targets.shape)
            token_counts = target_mask.sum(dim=1).clamp_min(1)
            mean_nll = (losses * target_mask).sum(dim=1) / token_counts
            ppls = torch.exp(mean_nll.clamp(max=20)).detach().cpu().tolist()
        return [float(value) for value in ppls]

    # Batched generation helper
    def generate_batch(curr_batch_size: int, seed_val: int, temp: float = 1.0, top_p: float = 0.9):
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)

        B = curr_batch_size
        input_ids = torch.full((B, 1), start_id, dtype=torch.long, device=device)
        cur_input = input_ids
        past_key_values = None

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated_tokens = [[] for _ in range(B)]
        n_res = torch.zeros(B, dtype=torch.long, device=device)

        with torch.no_grad():
            for step in range(MAX_LENGTH):
                outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]  # Shape: (B, vocab_size)
                past_key_values = outputs.past_key_values

                logits = logits / max(temp, 1e-5)

                # Mask non-canonical amino acids
                masked_logits = torch.full_like(logits, float("-inf"))
                masked_logits[:, standard_aa_ids] = logits[:, standard_aa_ids]

                # Allow end token only for sequences with length >= MIN_LENGTH
                allow_end_mask = n_res >= MIN_LENGTH
                masked_logits[allow_end_mask, end_id] = logits[allow_end_mask, end_id]

                # Nucleus sampling across batch
                probs = torch.softmax(masked_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                # CUDA cumsum has no deterministic implementation in supported PyTorch
                # releases. Compute this small vocabulary-axis reduction on CPU so the
                # same checkpoint/seed/runtime yields repeatable top-p masks.
                cumulative = torch.cumsum(sorted_probs.float().cpu(), dim=-1).to(device)

                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                next_tokens = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))  # (B, 1)

                # Record tokens
                for b in range(B):
                    if not finished[b]:
                        tok = next_tokens[b, 0].item()
                        if tok == end_id:
                            finished[b] = True
                        else:
                            generated_tokens[b].append(tok)
                            n_res[b] += 1

                if finished.all():
                    break

                cur_input = next_tokens

        # Decode batch
        decoded_seqs = []
        for b in range(B):
            seq = tokenizer.decode(generated_tokens[b]).replace(" ", "").strip()
            decoded_seqs.append(seq)
        return decoded_seqs

    # Generate total targets across temperature tiers
    temp_tiers = [
        (int(n_generate * 0.35), 0.85),
        (int(n_generate * 0.40), 1.00),
        (n_generate - int(n_generate * 0.35) - int(n_generate * 0.40), 1.15),
    ]

    all_candidates = []
    t_start = time.perf_counter()
    cand_idx = 0

    for tier_count, temp in temp_tiers:
        print(f"[*] Generating {tier_count} candidates at Temperature={temp} (batch_size={batch_size})...")
        generated_in_tier = 0

        while generated_in_tier < tier_count:
            curr_b = min(batch_size, tier_count - generated_in_tier)
            batch_seed = base_seed + cand_idx
            batch_seqs = generate_batch(curr_b, seed_val=batch_seed, temp=temp, top_p=0.90)
            batch_ppls = score_perplexity_batch(batch_seqs)

            for b_i, seq in enumerate(batch_seqs):
                is_canonical = set(seq).issubset(STANDARD_AAS)
                len_ok = MIN_LENGTH <= len(seq) <= MAX_LENGTH
                is_valid = is_canonical and len_ok and len(seq) > 0

                reason = "ok" if is_valid else ("invalid_residue" if not is_canonical else "invalid_length")

                all_candidates.append({
                    "run_id": f"ar_beam_{cand_idx:05d}",
                    "sequence": seq,
                    "length": len(seq),
                    "temperature": temp,
                    "top_p": 0.90,
                    "batch_seed": batch_seed,
                    "sample_index_in_batch": b_i,
                    "perplexity": round(batch_ppls[b_i], 4),
                    "passes_perplexity_filter": (
                        batch_ppls[b_i] <= max_perplexity if max_perplexity is not None else None
                    ),
                    "is_valid": is_valid,
                    "reason": reason,
                    "model": "progen2-small-epoch1-best",
                })
                cand_idx += 1

            generated_in_tier += curr_b
            elapsed = time.perf_counter() - t_start
            rate = cand_idx / max(elapsed, 0.01)
            print(f"  Progress: {cand_idx}/{n_generate} ({rate:.1f} seq/s, elapsed: {elapsed:.1f}s)")

    total_gen_time = time.perf_counter() - t_start
    print(f"✓ Generation finished: {len(all_candidates)} sequences in {total_gen_time:.1f}s ({(len(all_candidates)/total_gen_time):.1f} seq/s)")

    valid_seqs = [c for c in all_candidates if c["is_valid"]]
    unique_valid = set(c["sequence"] for c in valid_seqs)
    mean_length = sum(c["length"] for c in valid_seqs) / max(len(valid_seqs), 1)
    ppl_passes = [c["passes_perplexity_filter"] for c in all_candidates]

    return {
        "status": "success",
        "gpu": gpu_name,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "model_code_revision": PROGEN2_MODEL_REVISION,
        "max_perplexity": max_perplexity,
        "n_generated": len(all_candidates),
        "n_valid": len(valid_seqs),
        "n_unique": len(unique_valid),
        "n_perplexity_pass": (
            sum(flag is True for flag in ppl_passes) if max_perplexity is not None else None
        ),
        "validity_rate": round(len(valid_seqs) / len(all_candidates), 4),
        "mean_length": round(mean_length, 2),
        "mean_perplexity": round(
            sum(c["perplexity"] for c in all_candidates) / len(all_candidates), 4
        ),
        "generation_time_sec": round(total_gen_time, 2),
        "generation_rate_seq_per_sec": round(len(all_candidates) / total_gen_time, 1),
        "candidates": all_candidates,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate ProGen2 AMP candidates on Beam.")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        choices=["progen2_small_amp_best_val", "progen2_small_amp_finetuned"],
        default="progen2_small_amp_best_val",
    )
    parser.add_argument("--max-perplexity", type=float, default=None)
    args = parser.parse_args()

    print("=======================================================")
    print(" High-Speed Batched Autoregressive Sampling on Beam Cloud")
    print(f" Target: {args.n} sequences (batch size: {args.batch_size})")
    print(f" Checkpoint: /models/{args.checkpoint}")
    print("=======================================================\n")

    t0 = time.perf_counter()
    result = generate_candidates_batched.remote(
        n_generate=args.n,
        batch_size=args.batch_size,
        base_seed=args.seed,
        checkpoint_name=args.checkpoint,
        max_perplexity=args.max_perplexity,
    )
    total_elapsed = time.perf_counter() - t0

    print(f"\n✓ Completed in {total_elapsed:.1f}s ({result['generation_rate_seq_per_sec']} seq/s on {result['gpu']})")
    print(f"Generated: {result['n_generated']} sequences")
    print(f"Valid:     {result['n_valid']} ({result['validity_rate']*100:.1f}%)")
    print(f"Unique:    {result['n_unique']}")
    print(f"Mean Len:  {result['mean_length']} residues")

    # Local file destinations
    out_dir_ar = root / "autoregressive-models/outputs"
    out_dir_main = root / "outputs"

    out_dir_ar.mkdir(parents=True, exist_ok=True)
    out_dir_main.mkdir(parents=True, exist_ok=True)

    cand_df = pd.DataFrame(result["candidates"])

    cand_path_ar = out_dir_ar / "finetuned_candidates.csv"
    cand_path_main = out_dir_main / "ar_candidates.csv"
    metrics_path = out_dir_ar / "finetuned_metrics.json"

    cand_df.to_csv(cand_path_ar, index=False)
    if args.max_perplexity is not None:
        cand_df = cand_df[cand_df["passes_perplexity_filter"]].copy()
    if cand_df.empty:
        raise RuntimeError("No candidates passed the requested perplexity threshold")
    cand_df.to_csv(cand_path_main, index=False)
    print(f"\nSaved candidates to:")
    print(f"  - {cand_path_ar}")
    print(f"  - {cand_path_main}")

    metrics = {
        "model": "progen2-small-epoch1-best",
        "gpu": result["gpu"],
        "checkpoint": f"/models/{result['checkpoint_name']}",
        "checkpoint_sha256": result["checkpoint_sha256"],
        "model_code_revision": result["model_code_revision"],
        "torch_version": result["torch_version"],
        "transformers_version": result["transformers_version"],
        "seed": args.seed,
        "max_perplexity": result["max_perplexity"],
        "n_generated": result["n_generated"],
        "n_valid": result["n_valid"],
        "n_unique": result["n_unique"],
        "n_perplexity_pass": result["n_perplexity_pass"],
        "validity_rate": result["validity_rate"],
        "mean_length": result["mean_length"],
        "mean_perplexity": result["mean_perplexity"],
        "generation_time_sec": result["generation_time_sec"],
        "generation_rate_seq_per_sec": result["generation_rate_seq_per_sec"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")



if __name__ == "__main__":
    main()
