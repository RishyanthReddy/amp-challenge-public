"""
Generate a small pool from a fine-tuned checkpoint (baseline or challenger) and compare
it against the pretrained baseline's smoke test. Reads the tokenizer from inside the
checkpoint folder itself (saved there by 05_finetune.py), so this works automatically
for any model size without needing a separate --model-name flag.

Usage:
    python scripts/06_generate_from_checkpoint.py --checkpoint outputs/checkpoints/default/epoch_1 --n 100
    python scripts/06_generate_from_checkpoint.py --checkpoint outputs/checkpoints/medium/epoch_1 --n 100 --tag medium
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from common.model_utils import (
    generate_sequence,
    compute_perplexity,
    MODEL_NAME,
    MODEL_REVISION,
)
from common.validity import is_valid_sequence, summarize_validity


def load_finetuned(checkpoint_dir: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        code_revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    local_tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if os.path.exists(local_tokenizer_path):
        tokenizer = Tokenizer.from_file(local_tokenizer_path)
    else:
        print(f"No tokenizer.json in {checkpoint_dir}, falling back to {MODEL_NAME} from the Hub.")
        tokenizer = Tokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer.no_padding()
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to a saved epoch_N checkpoint dir")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(ROOT) / "outputs")
    parser.add_argument("--max-perplexity", type=float, default=None,
                        help="Filter candidates exceeding this perplexity threshold")
    parser.add_argument("--tag", default="",
                        help="Suffix for output filenames, e.g. 'medium' -> finetuned_candidates_medium.csv")
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be greater than zero")
    if args.max_perplexity is not None and (not math.isfinite(args.max_perplexity) or args.max_perplexity <= 0):
        parser.error("--max-perplexity must be greater than zero")
    suffix = f"_{args.tag}" if args.tag else ""

    model, tokenizer, device = load_finetuned(args.checkpoint, args.device)

    records = []
    for i in range(args.n):
        seed = args.seed + i
        seq = generate_sequence(model, tokenizer, device, seed=seed)
        ppl = compute_perplexity(model, tokenizer, seq, device) if seq else None
        if args.max_perplexity is not None and (ppl is None or not math.isfinite(ppl) or ppl > args.max_perplexity):
            continue
        valid, reason = is_valid_sequence(seq)
        records.append({
            "run_id": f"ft_{i:03d}",
            "sequence": seq,
            "length": len(seq),
            "perplexity": round(ppl, 2) if ppl is not None else None,
            "valid": valid,
            "reason": reason,
            "checkpoint": args.checkpoint,
            "seed": seed,
        })
        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{args.n}")

    if not records:
        raise RuntimeError(
            "No candidates passed generation/perplexity screening. "
            "Inspect the checkpoint and sampling settings before changing the threshold."
        )

    df = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"finetuned_candidates{suffix}.csv"
    df.to_csv(out_csv, index=False)

    metrics = summarize_validity(df["sequence"].tolist())
    metrics["mean_length"] = round(float(df["length"].mean()), 2)
    metrics["checkpoint"] = args.checkpoint

    out_json = args.output_dir / f"finetuned_metrics{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Fine-tuned generation summary ---")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
