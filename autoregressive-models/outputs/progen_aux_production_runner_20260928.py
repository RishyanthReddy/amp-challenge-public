"""Sample a reproducible auxiliary ProGen2 pool from the saved fine-tuned checkpoint.

Run with the Beam-enabled Python environment, for example::

    python cloud/run_progen_auxiliary.py --n-attempts 512 --run-id benchmark_a

The Beam function writes raw attempts to the private amp-models volume. The local driver
downloads the CSV, checks its hash, and writes a run manifest. No existing role output is
overwritten. A separate audit must deduplicate, apply reference checks, and select the
final library; raw sampling alone is not a submission artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from beam import Image, Volume, function


ROOT = Path(os.environ.get("AMP_CHALLENGE_ROOT", Path(__file__).resolve().parents[1]))
MODEL_REVISION = "43237a0b733c6629226a079266d2985c9fdce9b7"
CHECKPOINT_SHA256 = "124b8ea7df5c96cda927bada51c9d26d89f91636d0975fe70e7bc0ef182f9f83"
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 8
MAX_LENGTH = 50
TEMPERATURES = (0.85, 1.00, 1.15)
TOP_P = 0.90

image = Image(
    python_version="python3.10",
    python_packages=[
        "torch==2.5.1",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
    ],
)
models_volume = Volume(name="amp-models", mount_path="/models")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@function(
    gpu="RTX4090",
    image=image,
    memory="24Gi",
    cpu=8,
    volumes=[models_volume],
    timeout=3600,
)
def sample_auxiliary(n_attempts: int, base_seed: int, batch_size: int, run_id: str) -> dict:
    import csv
    import os
    import time
    import warnings

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    if not 1 <= n_attempts <= 100_000:
        raise ValueError("n_attempts must be between 1 and 100,000")
    if not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be between 1 and 256")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must contain only letters, digits, underscores, and hyphens")
    if not torch.cuda.is_available():
        raise RuntimeError("ProGen2 auxiliary run requires CUDA for the declared GPU workflow")

    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings(
        "ignore", message="cumsum_cuda_kernel does not have a deterministic implementation"
    )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    checkpoint_dir = Path("/models/progen2_small_amp_best_val")
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.is_file() or sha256_file(weight_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Beam ProGen2 checkpoint is missing or differs from the pinned SHA-256")
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")

    output_dir = Path("/models/progen2_auxiliary")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"progen_aux_{run_id}.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite a prior run: {output_path}")

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        code_revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    vocab = tokenizer.get_vocab()
    standard_ids = torch.tensor([vocab[aa] for aa in STANDARD_AA], device=device)
    start_id = tokenizer.encode("1").ids[0]
    end_id = tokenizer.encode("2").ids[0]

    def generate_batch(count: int, seed: int, temperature: float) -> list[str]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        current = torch.full((count, 1), start_id, dtype=torch.long, device=device)
        past = None
        finished = torch.zeros(count, dtype=torch.bool, device=device)
        lengths = torch.zeros(count, dtype=torch.long, device=device)
        tokens: list[list[int]] = [[] for _ in range(count)]
        with torch.no_grad():
            for _ in range(MAX_LENGTH):
                outputs = model(current, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                logits = outputs.logits[:, -1, :] / temperature
                masked = torch.full_like(logits, float("-inf"))
                masked[:, standard_ids] = logits[:, standard_ids]
                allow_end = lengths >= MIN_LENGTH
                masked[allow_end, end_id] = logits[allow_end, end_id]
                probabilities = torch.softmax(masked, dim=-1)
                sorted_probabilities, sorted_ids = torch.sort(
                    probabilities, descending=True, dim=-1
                )
                cumulative = torch.cumsum(sorted_probabilities, dim=-1)
                cutoff = cumulative > TOP_P
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probabilities[cutoff] = 0.0
                sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
                drawn = torch.multinomial(sorted_probabilities, num_samples=1)
                next_tokens = sorted_ids.gather(-1, drawn)
                for row in range(count):
                    if not finished[row]:
                        token = int(next_tokens[row, 0].item())
                        if token == end_id:
                            finished[row] = True
                        else:
                            tokens[row].append(token)
                            lengths[row] += 1
                if bool(finished.all()):
                    break
                current = next_tokens
        return [tokenizer.decode(row).replace(" ", "").strip() for row in tokens]

    started = time.perf_counter()
    unique_valid: set[str] = set()
    valid_count = 0
    with output_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "attempt_index", "sequence", "length", "temperature", "top_p",
                "batch_seed", "batch_offset", "is_valid",
            ),
        )
        writer.writeheader()
        attempted = 0
        while attempted < n_attempts:
            count = min(batch_size, n_attempts - attempted)
            temperature = TEMPERATURES[(attempted // batch_size) % len(TEMPERATURES)]
            seed = base_seed + attempted
            sequences = generate_batch(count, seed, temperature)
            if len(sequences) != count:
                raise RuntimeError("Sampler returned a different number of sequences")
            for offset, sequence in enumerate(sequences):
                is_valid = (
                    MIN_LENGTH <= len(sequence) <= MAX_LENGTH
                    and set(sequence).issubset(STANDARD_AA)
                )
                writer.writerow({
                    "attempt_index": attempted + offset + 1,
                    "sequence": sequence,
                    "length": len(sequence),
                    "temperature": temperature,
                    "top_p": TOP_P,
                    "batch_seed": seed,
                    "batch_offset": offset,
                    "is_valid": is_valid,
                })
                if is_valid:
                    valid_count += 1
                    unique_valid.add(sequence)
            attempted += count
            if attempted % 1024 == 0 or attempted == n_attempts:
                elapsed = time.perf_counter() - started
                print(f"Sampled {attempted}/{n_attempts} attempts ({attempted / elapsed:.1f}/s)")

    elapsed = time.perf_counter() - started
    return {
        "status": "success",
        "run_id": run_id,
        "n_attempts": n_attempts,
        "n_valid": valid_count,
        "n_unique_valid": len(unique_valid),
        "base_seed": base_seed,
        "batch_size": batch_size,
        "temperature_schedule": list(TEMPERATURES),
        "top_p": TOP_P,
        "max_length": MAX_LENGTH,
        "min_length": MIN_LENGTH,
        "elapsed_sec": round(elapsed, 3),
        "attempts_per_sec": round(n_attempts / elapsed, 3),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "model_code_revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "remote_csv": f"beam://amp-models/progen2_auxiliary/{output_path.name}",
        "csv_sha256": sha256_file(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-attempts", type=int, default=512)
    parser.add_argument("--base-seed", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not 1 <= args.n_attempts <= 100_000:
        parser.error("--n-attempts must be between 1 and 100,000")
    if not 1 <= args.batch_size <= 256:
        parser.error("--batch-size must be between 1 and 256")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id contains invalid characters")

    result = sample_auxiliary.remote(
        n_attempts=args.n_attempts,
        base_seed=args.base_seed,
        batch_size=args.batch_size,
        run_id=args.run_id,
    )
    if result.get("status") != "success":
        raise RuntimeError(f"Beam run failed: {result}")

    output_dir = ROOT / "autoregressive-models/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_csv = output_dir / f"progen_aux_{args.run_id}.csv"
    if local_csv.exists():
        raise FileExistsError(f"Refusing to overwrite a prior local run: {local_csv}")
    subprocess.run(
        ["beam", "cp", result["remote_csv"], str(local_csv.relative_to(ROOT))],
        check=True,
        cwd=ROOT,
    )
    if sha256_file(local_csv) != result["csv_sha256"]:
        raise RuntimeError(f"Downloaded CSV differs from Beam manifest: {local_csv}")
    result["local_csv"] = str(local_csv.relative_to(ROOT))
    manifest_path = output_dir / f"progen_aux_{args.run_id}.manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved {local_csv} and {manifest_path}")
    print(f"Valid: {result['n_valid']}/{result['n_attempts']}; unique valid: {result['n_unique_valid']}")
    print(f"GPU rate: {result['attempts_per_sec']} attempts/s; CSV SHA-256: {result['csv_sha256']}")


if __name__ == "__main__":
    main()
