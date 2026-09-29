"""
Role 02: Autoregressive Models — Best Checkpoint (Early Stopping) & Generation
Runs on Beam Cloud NVIDIA GeForce RTX 4090 (24GB VRAM)
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd

from beam import Image, Volume, function

PROGEN2_MODEL_REVISION = "43237a0b733c6629226a079266d2985c9fdce9b7"

# Remote container image
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

models_volume = Volume(name="amp-models", mount_path="/models")


@function(
    gpu=["RTX4090", "A100-80", "A10G"],
    image=image,
    memory="24Gi",
    cpu=8,
    volumes=[models_volume],
    timeout=1800,
)
def train_best_checkpoint_and_sample(
    train_sequences: list[str],
    val_sequences: list[str],
    max_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-5,
    n_generate: int = 3500,
    base_seed: int = 42,
) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModelForCausalLM
    from tokenizers import Tokenizer
    import math
    import copy
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

    print(f"[+] Loading base {MODEL_NAME} in bfloat16...")
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
    standard_aa_ids = torch.tensor([vocab[aa] for aa in STANDARD_AAS if aa in vocab], device=device)

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

    # 2. Optimizer
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

    # 3. Fine-Tuning with Early Stopping & Best Checkpoint Tracking
    print(f"\n--- Starting Fine-Tuning with Best Validation Tracking (max {max_epochs} epochs) ---")
    training_log = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_state_dict = None
    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(train_loader, is_train=True)
        val_loss = run_epoch(val_loader, is_train=False)
        dt = time.perf_counter() - t0

        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            # Keep copy of best weights on CPU to save memory
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        status_marker = "★ BEST" if is_best else ""
        print(f"Epoch {epoch}/{max_epochs} ({dt:.1f}s) | Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) {status_marker}")

        training_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "is_best": is_best,
            "elapsed_sec": round(dt, 2),
        })

    train_time = time.perf_counter() - t_start
    print(f"\n✓ Training completed. Selecting Best Model from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})...")
    # Restore best weights to GPU
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    del best_state_dict

    # Save best checkpoint to persistent Beam Volume
    ckpt_dir = "/models/progen2_small_amp_best_val"
    try:
        model.save_pretrained(ckpt_dir)
        tokenizer.save(f"{ckpt_dir}/tokenizer.json")
        print(f"✓ Saved best checkpoint (Epoch {best_epoch}) to volume: {ckpt_dir}")
    except Exception as e:
        print(f"! Warning: Failed saving to volume: {e}")

    # 4. Batched Autoregressive Sampling from Best Checkpoint
    print(f"\n--- Generating {n_generate} Candidates with Batched KV-Cached Sampling (B=64) ---")
    model.eval()

    def generate_batch(curr_b: int, seed_val: int, temp: float = 1.0, top_p: float = 0.9):
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)

        B = curr_b
        input_ids = torch.full((B, 1), start_id, dtype=torch.long, device=device)
        cur_input = input_ids
        past_key_values = None

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated_tokens = [[] for _ in range(B)]
        n_res = torch.zeros(B, dtype=torch.long, device=device)

        with torch.no_grad():
            for step in range(MAX_LENGTH):
                outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                logits = logits / max(temp, 1e-5)

                # Mask non-canonical amino acids
                masked_logits = torch.full_like(logits, float("-inf"))
                masked_logits[:, standard_aa_ids] = logits[:, standard_aa_ids]

                # Allow end token only when length >= MIN_LENGTH (8 residues)
                allow_end = n_res >= MIN_LENGTH
                masked_logits[allow_end, end_id] = logits[allow_end, end_id]

                # Nucleus sampling
                probs = torch.softmax(masked_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)

                cutoff = cumulative > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                next_tokens = sorted_idx.gather(-1, torch.multinomial(sorted_probs, num_samples=1))

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

        decoded_seqs = []
        for b in range(B):
            seq = tokenizer.decode(generated_tokens[b]).replace(" ", "").strip()
            decoded_seqs.append(seq)
        return decoded_seqs

    # Generate across temperature tiers
    temp_tiers = [
        (int(n_generate * 0.35), 0.85),
        (int(n_generate * 0.40), 1.00),
        (n_generate - int(n_generate * 0.35) - int(n_generate * 0.40), 1.15),
    ]

    all_candidates = []
    t_gen_start = time.perf_counter()
    cand_idx = 0
    batch_size_gen = 64

    for tier_count, temp in temp_tiers:
        print(f"[*] Sampling {tier_count} candidates at Temp={temp}...")
        tier_done = 0
        while tier_done < tier_count:
            curr_b = min(batch_size_gen, tier_count - tier_done)
            batch_seed = base_seed + cand_idx
            seqs = generate_batch(curr_b, seed_val=batch_seed, temp=temp, top_p=0.90)

            for b_i, seq in enumerate(seqs):
                is_canonical = set(seq).issubset(STANDARD_AAS)
                len_ok = MIN_LENGTH <= len(seq) <= MAX_LENGTH
                is_valid = is_canonical and len_ok and len(seq) > 0
                reason = "ok" if is_valid else ("invalid_residue" if not is_canonical else "invalid_length")

                all_candidates.append({
                    "run_id": f"ar_best_{cand_idx:05d}",
                    "sequence": seq,
                    "length": len(seq),
                    "temperature": temp,
                    "top_p": 0.90,
                    "seed": batch_seed + b_i,
                    "is_valid": is_valid,
                    "reason": reason,
                    "model": "progen2-small-epoch1-best",
                    "checkpoint_epoch": best_epoch,
                })
                cand_idx += 1

            tier_done += curr_b
            elapsed = time.perf_counter() - t_gen_start
            rate = cand_idx / max(elapsed, 0.01)
            print(f"  Progress: {cand_idx}/{n_generate} ({rate:.1f} seq/s)")

    total_gen_time = time.perf_counter() - t_gen_start
    print(f"✓ Generated {len(all_candidates)} candidates in {total_gen_time:.1f}s ({(len(all_candidates)/total_gen_time):.1f} seq/s)")

    valid_seqs = [c for c in all_candidates if c["is_valid"]]
    unique_valid = set(c["sequence"] for c in valid_seqs)
    mean_length = sum(c["length"] for c in valid_seqs) / max(len(valid_seqs), 1)

    return {
        "status": "success",
        "gpu": gpu_name,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "train_time_sec": round(train_time, 2),
        "gen_time_sec": round(total_gen_time, 2),
        "training_log": training_log,
        "n_generated": len(all_candidates),
        "n_valid": len(valid_seqs),
        "n_unique": len(unique_valid),
        "validity_rate": round(len(valid_seqs) / len(all_candidates), 4),
        "mean_length": round(mean_length, 2),
        "candidates": all_candidates,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parquet_path = root / "data-engineering/data/processed/views/autoregressive_view.parquet"

    print(f"Loading AR dataset view from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    train_splits = {"core_train_only", "train"}
    train_df = df[df["split"].isin(train_splits)]
    val_df = df[df["split"] == "validation"]

    train_seqs = train_df["sequence"].dropna().tolist()
    val_seqs = val_df["sequence"].dropna().tolist()

    print(f"Training split: {len(train_seqs)} sequences")
    print(f"Validation split: {len(val_seqs)} sequences")

    N_GENERATE = 3500
    MAX_EPOCHS = 3
    BATCH_SIZE = 32

    print(f"\n=======================================================")
    print(f" Executing Best-Checkpoint AR Run on Beam Cloud RTX 4090")
    print(f" Early Stopping / Best Model Selection enabled")
    print(f" Target Generation: {N_GENERATE} diverse candidates")
    print(f"=======================================================\n")

    t_total_start = time.perf_counter()
    result = train_best_checkpoint_and_sample.remote(
        train_sequences=train_seqs,
        val_sequences=val_seqs,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=5e-5,
        n_generate=N_GENERATE,
        base_seed=42,
    )

    total_time = time.perf_counter() - t_total_start
    print(f"\n✓ Completed on {result['gpu']} in {total_time:.1f}s")
    print(f"Selected Best Checkpoint: Epoch {result['best_epoch']} (Val Loss: {result['best_val_loss']})")
    print(f"Training Time: {result['train_time_sec']}s")
    print(f"Generation Time: {result['gen_time_sec']}s ({result['n_generated']} seqs)")
    print(f"Valid Sequences: {result['n_valid']} ({result['validity_rate']*100:.1f}%)")
    print(f"Unique Valid:    {result['n_unique']}")
    print(f"Mean Length:     {result['mean_length']} residues")

    # Local file destinations
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
    print(f"Saved candidates to {cand_path_ar} and {cand_path_main}")

    metrics = {
        "model": "progen2-small-epoch1-best",
        "selected_best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "checkpoint": "/models/progen2_small_amp_best_val",
        "n_generated": result["n_generated"],
        "n_valid": result["n_valid"],
        "n_unique": result["n_unique"],
        "validity_rate": result["validity_rate"],
        "mean_length": result["mean_length"],
        "generation_time_sec": result["gen_time_sec"],
        "train_time_sec": result["train_time_sec"],
        "training_log": result["training_log"],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    pd.DataFrame(result["training_log"]).to_csv(log_path, index=False)
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
