"""
Beam Cloud Training & Generation Runner for Role 02: Autoregressive Models
Hardware: Cloud NVIDIA GeForce RTX 4090 (24GB VRAM)
Model: hugohrban/progen2-small (151M parameters)
Data: 26,699 AMP sequences from autoregressive_view.parquet
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd

from beam import Image, Volume, function

PROGEN2_MODEL_REVISION = "43237a0b733c6629226a079266d2985c9fdce9b7"

# Remote container image specification with automatic CUDA torch
image = Image(
    python_version="python3.10",
    python_packages=[
        "torch",
        "transformers==4.46.3",
        "tokenizers",
        "numpy",
        "pandas",
        "accelerate",
    ],
)

# Persistent volume for checkpoints
models_volume = Volume(name="amp-models", mount_path="/models")


@function(
    gpu=["RTX4090", "A100-80", "A10G"],
    image=image,
    memory="24Gi",
    cpu=8,
    volumes=[models_volume],
    timeout=3600,
)
def fine_tune_and_generate_progen2(
    train_sequences: list[str],
    val_sequences: list[str],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    n_generate: int = 2500,
    base_seed: int = 42,
) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM
    from tokenizers import Tokenizer
    import math
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Initializing on {gpu_name} (PyTorch {torch.__version__}, CUDA {torch.version.cuda})")

    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    MODEL_NAME = "hugohrban/progen2-small"
    START_TOKEN = "1"
    END_TOKEN = "2"
    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    MIN_LENGTH = 8
    MAX_LENGTH = 50

    print(f"[+] Loading {MODEL_NAME} in bfloat16...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=PROGEN2_MODEL_REVISION,
        code_revision=PROGEN2_MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)

    tokenizer = Tokenizer.from_pretrained(MODEL_NAME, revision=PROGEN2_MODEL_REVISION)
    tokenizer.no_padding()

    start_id = tokenizer.encode(START_TOKEN).ids[0]
    end_id = tokenizer.encode(END_TOKEN).ids[0]
    vocab = tokenizer.get_vocab()
    standard_aa_ids = [vocab[aa] for aa in STANDARD_AAS if aa in vocab]
    print(f"[+] Tokenizer ready: {len(standard_aa_ids)} standard AA tokens mapped.")

    # 1. Dataset & Collate
    class SequenceDataset(Dataset):
        def __init__(self, seqs, max_len=52):
            self.examples = []
            for s in seqs:
                token_ids = tokenizer.encode(START_TOKEN + s + END_TOKEN).ids
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                self.examples.append(token_ids)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    def make_collate_fn(pad_token_id):
        def collate(batch):
            max_len = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, ids in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, : len(ids)] = 1
            return input_ids, attention_mask
        return collate

    train_ds = SequenceDataset(train_sequences)
    val_ds = SequenceDataset(val_sequences)
    collate = make_collate_fn(end_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    print(f"[+] DataLoaders created: {len(train_ds)} train examples ({len(train_loader)} batches), {len(val_ds)} val examples.")

    # 2. Optimizer & Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    def run_epoch(loader, is_train=True):
        if is_train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_tokens = 0

        context_mgr = torch.enable_grad() if is_train else torch.no_grad()
        with context_mgr:
            for step, (input_ids, attention_mask) in enumerate(loader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                vocab_size = logits.size(-1)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous().float()

                flat_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                flat_loss = flat_loss.view(shift_labels.shape)
                token_count = shift_mask.sum().item()
                loss = (flat_loss * shift_mask).sum() / max(token_count, 1.0)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * token_count
                total_tokens += token_count

        return total_loss / max(total_tokens, 1)

    # 3. Fine-Tuning Loop
    print(f"\n--- Starting Fine-Tuning ({epochs} epochs, lr={lr}) ---")
    training_log = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(train_loader, is_train=True)
        val_loss = run_epoch(val_loader, is_train=False)
        dt = time.perf_counter() - t0

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        print(f"Epoch {epoch}/{epochs} ({dt:.1f}s) | Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")
        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "elapsed_sec": round(dt, 2),
        })

    train_time = time.perf_counter() - t_start
    print(f"✓ Fine-tuning completed in {train_time:.1f}s")

    # Save model checkpoint to Beam Volume
    ckpt_dir = "/models/progen2_small_amp_finetuned"
    try:
        model.save_pretrained(ckpt_dir)
        tokenizer.save(f"{ckpt_dir}/tokenizer.json")
        print(f"✓ Saved checkpoint to volume: {ckpt_dir}")
    except Exception as e:
        print(f"! Warning: Failed saving to volume: {e}")

    # 4. High-Throughput Candidate Generation
    print(f"\n--- Generating {n_generate} AMP Candidates with KV Caching & Nucleus Sampling ---")
    model.eval()

    def generate_single_seq(seed_val, temp=1.0, top_p=0.9):
        torch.manual_seed(seed_val)
        input_ids = torch.tensor([[start_id]], device=device)
        cur_input = input_ids
        past_key_values = None
        n_res = 0
        collected_tokens = []

        with torch.no_grad():
            for _ in range(MAX_LENGTH):
                outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                logits = logits / max(temp, 1e-5)

                allowed = list(standard_aa_ids)
                if n_res >= MIN_LENGTH:
                    allowed.append(end_id)

                masked_logits = torch.full_like(logits, float("-inf"))
                masked_logits[:, allowed] = logits[:, allowed]

                probs = torch.softmax(masked_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)

                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))

                token_val = next_token.item()
                if token_val == end_id:
                    break

                collected_tokens.append(token_val)
                n_res += 1
                cur_input = next_token

        decoded = tokenizer.decode(collected_tokens).replace(" ", "").strip()
        return decoded

    candidates = []
    t_gen_start = time.perf_counter()

    # Temperature variations for exploration
    temperatures = [0.85, 1.0, 1.15]

    for i in range(n_generate):
        s_seed = base_seed + i
        temp = temperatures[i % len(temperatures)]
        seq = generate_single_seq(s_seed, temp=temp, top_p=0.90)

        # Basic validity check
        is_canonical = set(seq).issubset(STANDARD_AAS)
        len_valid = MIN_LENGTH <= len(seq) <= MAX_LENGTH
        is_valid = is_canonical and len_valid and len(seq) > 0

        candidates.append({
            "run_id": f"ar_beam_{i:05d}",
            "sequence": seq,
            "length": len(seq),
            "temperature": temp,
            "top_p": 0.90,
            "seed": s_seed,
            "is_valid": is_valid,
            "model": "progen2-small-finetuned",
        })

        if (i + 1) % 500 == 0 or i + 1 == n_generate:
            elapsed = time.perf_counter() - t_gen_start
            rate = (i + 1) / elapsed
            print(f"  Generated {i + 1}/{n_generate} sequences ({rate:.1f} seq/s, elapsed: {elapsed:.1f}s)")

    gen_time = time.perf_counter() - t_gen_start
    print(f"✓ Generated {len(candidates)} sequences in {gen_time:.1f}s ({(len(candidates)/gen_time):.1f} seq/s)")

    # 5. Compute summary statistics
    valid_count = sum(1 for c in candidates if c["is_valid"])
    unique_count = len(set(c["sequence"] for c in candidates if c["is_valid"]))
    lengths = [c["length"] for c in candidates if c["is_valid"]]
    mean_len = sum(lengths) / max(len(lengths), 1)

    return {
        "status": "success",
        "gpu": gpu_name,
        "train_time_sec": round(train_time, 2),
        "gen_time_sec": round(gen_time, 2),
        "training_log": training_log,
        "n_generated": len(candidates),
        "n_valid": valid_count,
        "n_unique": unique_count,
        "mean_length": round(mean_len, 2),
        "candidates": candidates,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parquet_path = root / "data-engineering/data/processed/views/autoregressive_view.parquet"

    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist!")
        sys.exit(1)

    print(f"Loading AR dataset view from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} total sequences.")

    train_splits = {"core_train_only", "train"}
    train_df = df[df["split"].isin(train_splits)]
    val_df = df[df["split"] == "validation"]

    train_seqs = train_df["sequence"].dropna().tolist()
    val_seqs = val_df["sequence"].dropna().tolist()

    print(f"Training split: {len(train_seqs)} sequences")
    print(f"Validation split: {len(val_seqs)} sequences")

    # Target number of sequences to generate
    N_GENERATE = 3000
    EPOCHS = 3
    BATCH_SIZE = 32

    print(f"\n=======================================================")
    print(f" Dispatching Fine-Tuning & Generation to Beam Cloud GPU")
    print(f" Target Hardware: NVIDIA GeForce RTX 4090")
    print(f" Training: {len(train_seqs)} sequences, {EPOCHS} epochs, batch_size={BATCH_SIZE}")
    print(f" Generation: {N_GENERATE} diverse candidates")
    print(f"=======================================================\n")

    t_total_start = time.perf_counter()
    result = fine_tune_and_generate_progen2.remote(
        train_sequences=train_seqs,
        val_sequences=val_seqs,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=5e-5,
        n_generate=N_GENERATE,
        base_seed=42,
    )

    total_time = time.perf_counter() - t_total_start
    print(f"\n✓ Remote execution completed in {total_time:.1f}s")
    print(f"GPU Worker: {result['gpu']}")
    print(f"Training Time: {result['train_time_sec']}s")
    print(f"Generation Time: {result['gen_time_sec']}s ({result['n_generated']} seqs)")
    print(f"Valid Sequences: {result['n_valid']} / {result['n_generated']}")
    print(f"Unique Valid Sequences: {result['n_unique']}")
    print(f"Mean Length: {result['mean_length']} residues")

    # Save to local destinations
    out_dir_ar = root / "autoregressive-models/outputs"
    out_dir_main = root / "outputs"
    docs_dir = root / "autoregressive-models/docs"

    out_dir_ar.mkdir(parents=True, exist_ok=True)
    out_dir_main.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    cand_df = pd.DataFrame(result["candidates"])

    cand_path_ar = out_dir_ar / "finetuned_candidates.csv"
    cand_path_main = out_dir_main / "ar_candidates.csv"
    metrics_path = out_dir_ar / "finetuned_metrics.json"
    log_path = docs_dir / "finetune_log.csv"

    cand_df.to_csv(cand_path_ar, index=False)
    cand_df.to_csv(cand_path_main, index=False)
    print(f"Saved candidate sequences to {cand_path_ar} and {cand_path_main}")

    metrics_payload = {
        "status": result["status"],
        "gpu": result["gpu"],
        "train_time_sec": result["train_time_sec"],
        "gen_time_sec": result["gen_time_sec"],
        "total_elapsed_sec": round(total_time, 2),
        "n_generated": result["n_generated"],
        "n_valid": result["n_valid"],
        "n_unique": result["n_unique"],
        "mean_length": result["mean_length"],
        "training_log": result["training_log"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    log_df = pd.DataFrame(result["training_log"])
    log_df.to_csv(log_path, index=False)
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
