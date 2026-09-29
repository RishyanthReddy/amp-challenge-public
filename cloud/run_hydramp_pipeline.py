"""
Role 03: HydrAMP VAE Dual-Mode Generation & Calibrated Radius Sweep on Beam Cloud
Executes:
1. Mode 1: Unconstrained Sampling (Prior z ~ N(0, I), condition y=[1, 1])
2. Mode 2: Analogue Perturbation across 60 eligible seeds with temperature sweep (temp in [1.0, 2.0, 3.0, 4.0])
3. Detailed provenance tracking (parent_sequence_id, parent_sequence, temp, amp_prob, mic_prob)
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
import pandas as pd

from beam import Image, Volume, function

# Pinned environment matching HydrAMP specification
image = Image(
    python_version="python3.8",
    commands=[
        "pip install protobuf==3.14.0 numpy==1.18.5",
        "pip install tensorflow==2.2.1 Keras==2.3.1 tensorflow-probability==0.10.1",
        "pip install joblib==0.17.0 scikit-learn==0.23.2 pandas==1.1.4 matplotlib==3.3.2",
        "pip install tqdm==4.66.6 biopython==1.83 Levenshtein==0.25.0",
        "pip install modlamp==4.2.3 --no-deps",
        "pip install git+https://github.com/szczurek-lab/hydramp.git@6590d2f4c2963f25d30669052a4c4a857e0e7279 --no-deps",
    ],
)

models_volume = Volume(name="amp-models", mount_path="/models")
HYDRAMP_UPSTREAM_REVISION = "6590d2f4c2963f25d30669052a4c4a857e0e7279"


def sha256_tree(root: Path) -> str:
    """Hash relative paths and file contents so checkpoint layout is covered."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@function(
    gpu=["RTX4090", "A10G"],
    image=image,
    memory="24Gi",
    cpu=8,
    volumes=[models_volume],
    timeout=1800,
)
def run_hydramp_generation_and_sweep(
    seed_records: list,  # list of dicts: {'seed_id': ..., 'sequence': ..., 'cluster_id': ...}
    n_unconstrained: int = 1000,
    n_analogue_per_seed: int = 20,
    base_seed: int = 42,
) -> dict:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    import keras
    import joblib
    import numpy as np
    import time
    from amp.inference.inference import HydrAMPGenerator

    print(f"🚀 Worker starting on Python {sys.version}")
    gpu_name = tf.test.gpu_device_name() or "CPU"
    print(f"🎮 TensorFlow device: {gpu_name}")

    model_path = "/models/hydramp/model"
    decomposer_path = "/models/hydramp/pca_decomposer.joblib"
    model_tree_sha256 = sha256_tree(Path(model_path))
    decomposer_sha256 = hashlib.sha256(Path(decomposer_path).read_bytes()).hexdigest()

    print(f"[+] Loading HydrAMPGenerator from {model_path}...")
    t0 = time.time()
    generator = HydrAMPGenerator(
        model_path=model_path,
        decomposer_path=decomposer_path,
        softmax=True,
    )
    print(f"✓ Generator loaded in {time.time() - t0:.2f}s")

    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

    # =========================================================================
    # 1. Mode 1: Unconstrained Sampling
    # =========================================================================
    print(f"\n--- Generating {n_unconstrained} Mode 1 (Unconstrained) Candidates ---")
    t_unconstrained = time.time()
    
    unconstrained_raw = generator.unconstrained_generation(
        mode="amp",
        n_target=n_unconstrained,
        seed=base_seed,
        filter_out=True,
        properties=True,
        n_attempts=1,
    )
    dt_unconstrained = time.time() - t_unconstrained
    print(f"✓ Mode 1 generated {len(unconstrained_raw)} sequences in {dt_unconstrained:.2f}s ({(len(unconstrained_raw)/max(dt_unconstrained, 0.1)):.1f} seq/s)")

    unconstrained_records = []
    for i, r in enumerate(unconstrained_raw):
        seq = str(r["sequence"]).strip().upper()
        is_canonical = set(seq).issubset(STANDARD_AAS)
        len_ok = 8 <= len(seq) <= 25
        is_valid = is_canonical and len_ok and len(seq) > 0
        reason = "ok" if is_valid else ("invalid_residue" if not is_canonical else "invalid_length")

        unconstrained_records.append({
            "run_id": f"vae_unconstrained_{i:05d}",
            "sequence": seq,
            "length": len(seq),
            "generation_mode": "unconstrained",
            "parent_sequence_id": None,
            "parent_sequence": None,
            "prototype_cluster_id": None,
            "perturbation_temp": None,
            "amp_prob": round(float(r.get("amp", 0)), 4),
            "mic_prob": round(float(r.get("mic", 0)), 4),
            "is_valid": is_valid,
            "reason": reason,
            "seed": base_seed,
            "sample_index": i,
            "model": "HydrAMP_epoch37",
        })

    # =========================================================================
    # 2. Mode 2: Analogue Perturbation & Radius Sweep
    # =========================================================================
    # Sweep across temperatures: 1.0 (conservative), 2.0 (moderate), 3.0 (exploratory), 4.0 (wide)
    temp_sweep = [1.0, 2.0, 3.0, 4.0]
    print(f"\n--- Generating Mode 2 (Analogues) across {len(seed_records)} Seeds and Temperatures {temp_sweep} ---")
    t_analogue_total = time.time()

    analogue_records = []
    sweep_stats = {t: {"attempts": 0, "produced": 0, "valid": 0} for t in temp_sweep}

    seed_seq_to_meta = {r["sequence"]: r for r in seed_records}
    seeds_list = [r["sequence"] for r in seed_records]

    for temp_val in temp_sweep:
        t_temp = time.time()
        print(f"[*] Running Analogue Perturbation at Temp={temp_val} (attempts per seed={n_analogue_per_seed})...")
        
        analogue_batch = generator.analogue_generation(
            sequences=seeds_list,
            seed=base_seed + int(temp_val * 100),
            filtering_criteria="discovery",
            n_attempts=n_analogue_per_seed,
            temp=temp_val,
        )

        for parent_seq, p_data in analogue_batch.items():
            meta = seed_seq_to_meta.get(parent_seq, {})
            children = p_data.get("generated_sequences", [])
            sweep_stats[temp_val]["attempts"] += n_analogue_per_seed
            if children:
                for c in children:
                    c_seq = str(c["sequence"]).strip().upper()
                    if c_seq == parent_seq:
                        continue  # discard identical parent echo

                    is_canonical = set(c_seq).issubset(STANDARD_AAS)
                    len_ok = 8 <= len(c_seq) <= 25
                    is_valid = is_canonical and len_ok and len(c_seq) > 0
                    reason = "ok" if is_valid else ("invalid_residue" if not is_canonical else "invalid_length")

                    sweep_stats[temp_val]["produced"] += 1
                    if is_valid:
                        sweep_stats[temp_val]["valid"] += 1

                    analogue_records.append({
                        "run_id": f"vae_analogue_{len(analogue_records):05d}",
                        "sequence": c_seq,
                        "length": len(c_seq),
                        "generation_mode": "analogue",
                        "parent_sequence_id": meta.get("seed_id", "unknown_seed"),
                        "parent_sequence": parent_seq,
                        "prototype_cluster_id": meta.get("cluster_id", "unassigned"),
                        "perturbation_temp": temp_val,
                        "amp_prob": round(float(c.get("amp", 0)), 4),
                        "mic_prob": round(float(c.get("mic", 0)), 4),
                        "is_valid": is_valid,
                        "reason": reason,
                        "seed": base_seed + int(temp_val * 100),
                        "model": "HydrAMP_epoch37",
                    })

        dt_temp = time.time() - t_temp
        print(f"  Temp={temp_val}: produced {sweep_stats[temp_val]['produced']} analogues in {dt_temp:.2f}s")

    total_analogue_time = time.time() - t_analogue_total
    print(f"✓ Mode 2 complete: {len(analogue_records)} total analogues across sweep in {total_analogue_time:.2f}s")

    return {
        "status": "success",
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "device": gpu_name,
        "checkpoint_tree_sha256": model_tree_sha256,
        "pca_decomposer_sha256": decomposer_sha256,
        "upstream_revision": HYDRAMP_UPSTREAM_REVISION,
        "base_seed": base_seed,
        "n_unconstrained_requested": n_unconstrained,
        "n_analogue_per_seed_requested": n_analogue_per_seed,
        "unconstrained_records": unconstrained_records,
        "analogue_records": analogue_records,
        "sweep_stats": sweep_stats,
        "unconstrained_time_sec": round(dt_unconstrained, 2),
        "analogue_time_sec": round(total_analogue_time, 2),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    proto_csv = root / "vae-latent-models/outputs/prototype_panel_validated.csv"
    if not proto_csv.exists():
        print(f"Error: {proto_csv} not found!")
        sys.exit(1)

    proto_df = pd.read_csv(proto_csv)
    eligible_seeds = proto_df[proto_df["hydramp_eligible"]].to_dict("records")
    print(f"Loaded {len(eligible_seeds)} eligible seeds for analogue perturbation.")

    N_UNCONSTRAINED = 1000
    N_ANALOGUE_PER_SEED = 25  # 60 seeds * 25 attempts * 4 temps = 6,000 attempts

    print(f"\n=======================================================")
    print(f" Executing HydrAMP Dual-Mode Generation & Sweep on Beam Cloud")
    print(f" Mode 1: {N_UNCONSTRAINED} Unconstrained Candidates")
    print(f" Mode 2: {len(eligible_seeds)} Seeds x 4 Temperatures ([1.0, 2.0, 3.0, 4.0])")
    print(f"=======================================================\n")

    t0 = time.time()
    result = run_hydramp_generation_and_sweep.remote(
        seed_records=eligible_seeds,
        n_unconstrained=N_UNCONSTRAINED,
        n_analogue_per_seed=N_ANALOGUE_PER_SEED,
        base_seed=42,
    )
    total_time = time.time() - t0
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError("HydrAMP Beam generation failed; no candidate artifacts were written")
    print(f"\n✓ Remote generation completed in {total_time:.1f}s!")

    # 1. Unconstrained smoke file (Task 6 deliverable: first 100 attempts)
    unconstrained_df = pd.DataFrame(result["unconstrained_records"])
    smoke_unconstrained_csv = root / "vae-latent-models/outputs/unconstrained_smoke.csv"
    unconstrained_df.head(100).to_csv(smoke_unconstrained_csv, index=False)
    print(f"✓ Saved Mode 1 smoke test (100 attempts) to {smoke_unconstrained_csv}")

    # 2. Analogue smoke file (Task 8 deliverable: first 100 attempts with ancestry)
    analogue_df = pd.DataFrame(result["analogue_records"])
    smoke_analogue_csv = root / "vae-latent-models/outputs/analogue_smoke.csv"
    analogue_df.head(100).to_csv(smoke_analogue_csv, index=False)
    print(f"✓ Saved Mode 2 smoke test (100 attempts with ancestry) to {smoke_analogue_csv}")

    # 3. Radius sweep study (Task 10 deliverable)
    sweep_records = []
    for temp, s in result["sweep_stats"].items():
        sweep_records.append({
            "perturbation_temp": temp,
            "total_attempts": s["attempts"],
            "produced_analogues": s["produced"],
            "valid_analogues": s["valid"],
            "success_rate": round(s["produced"] / max(s["attempts"], 1), 4),
            "validity_rate": round(s["valid"] / max(s["produced"], 1), 4),
        })
    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv = root / "vae-latent-models/outputs/latent_setting_study.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"✓ Saved radius sweep study to {sweep_csv}")
    print("\n--- Radius Sweep Summary ---")
    print(sweep_df.to_string(index=False))

    # 4. Matched Mode Comparison table (Task 10 deliverable: 1,000 attempts per mode)
    # Take first 1,000 unconstrained and first 1,000 analogues
    matched_unconstrained = unconstrained_df.head(1000).copy()
    matched_analogues = analogue_df.head(1000).copy()

    comparison_records = [
        {
            "generation_mode": "unconstrained",
            "attempts": len(matched_unconstrained),
            "valid_sequences": int(matched_unconstrained["is_valid"].sum()),
            "unique_sequences": int(matched_unconstrained["sequence"].nunique()),
            "mean_length": round(float(matched_unconstrained["length"].mean()), 2),
            "mean_amp_prob": round(float(matched_unconstrained["amp_prob"].mean()), 4),
            "mean_mic_prob": round(float(matched_unconstrained["mic_prob"].mean()), 4),
        },
        {
            "generation_mode": "analogue",
            "attempts": len(matched_analogues),
            "valid_sequences": int(matched_analogues["is_valid"].sum()),
            "unique_sequences": int(matched_analogues["sequence"].nunique()),
            "mean_length": round(float(matched_analogues["length"].mean()), 2),
            "mean_amp_prob": round(float(matched_analogues["amp_prob"].mean()), 4),
            "mean_mic_prob": round(float(matched_analogues["mic_prob"].mean()), 4),
        }
    ]
    comp_df = pd.DataFrame(comparison_records)
    comp_csv = root / "vae-latent-models/outputs/vae_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\n✓ Saved matched mode comparison to {comp_csv}")
    print("\n--- Matched Mode Comparison (1,000 vs 1,000) ---")
    print(comp_df.to_string(index=False))

    # Also save the full raw pools for downstream filtering and scaling
    full_combined = pd.concat([unconstrained_df, analogue_df], ignore_index=True)
    full_combined["checkpoint_sha256"] = result["checkpoint_tree_sha256"]
    full_combined["decomposer_sha256"] = result["pca_decomposer_sha256"]
    full_combined["model_version"] = result["upstream_revision"]
    raw_combined_csv = root / "vae-latent-models/outputs/raw_generated_pool.csv"
    full_combined.to_csv(raw_combined_csv, index=False)
    print(f"✓ Saved full combined raw pool ({len(full_combined)} records) to {raw_combined_csv}")

    valid_mask = full_combined["is_valid"].fillna(False).astype(bool)
    candidate_pool = full_combined.loc[valid_mask].copy()
    candidate_pool["sequence"] = candidate_pool["sequence"].astype(str).str.strip().str.upper()
    candidate_pool = candidate_pool.drop_duplicates("sequence", keep="first").reset_index(drop=True)
    candidate_pool.insert(0, "sequence_id", candidate_pool["run_id"].astype(str))
    candidate_pool["domain"] = "vae_latent"
    candidate_pool["checkpoint_sha256"] = result["checkpoint_tree_sha256"]
    candidate_path = root / "outputs/vae_candidates.csv"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_pool.to_csv(candidate_path, index=False)

    manifest = {
        "role": "03_hydramp_generation",
        "upstream_revision": result["upstream_revision"],
        "checkpoint_tree_sha256": result["checkpoint_tree_sha256"],
        "pca_decomposer_sha256": result["pca_decomposer_sha256"],
        "tensorflow_version": result["tensorflow_version"],
        "keras_version": result["keras_version"],
        "device": result["device"],
        "base_seed": result["base_seed"],
        "n_unconstrained_requested": result["n_unconstrained_requested"],
        "n_unconstrained_returned": len(unconstrained_df),
        "n_analogue_per_seed_requested": result["n_analogue_per_seed_requested"],
        "analogue_seed_count": len(eligible_seeds),
        "raw_candidate_count": len(full_combined),
        "valid_unique_candidate_count": len(candidate_pool),
        "sweep_stats": result["sweep_stats"],
        "generation_wall_time_sec": round(total_time, 2),
        "candidate_csv_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }
    manifest_path = root / "vae-latent-models/outputs/hydramp_generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"✓ Saved {len(candidate_pool)} unique valid candidates to {candidate_path}")
    print(f"✓ Saved generation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
