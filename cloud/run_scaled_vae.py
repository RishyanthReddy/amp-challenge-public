"""
Role 03: Scaled Balanced VAE Generation on Beam Cloud RTX 4090
Generates:
- Mode 1: 2,000 Unconstrained Candidates (z ~ N(0, I), condition y=[1, 1])
- Mode 2: 2,000 Analogue Candidates across 60 seeds using calibrated temperatures (T=2.0, T=3.0)
Total: 4,000 candidates with strict ancestry tracking.
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd

from beam import Image, Volume, function

image = Image(
    python_version="python3.8",
    commands=[
        "pip install protobuf==3.14.0 numpy==1.18.5",
        "pip install tensorflow==2.2.1 Keras==2.3.1 tensorflow-probability==0.10.1",
        "pip install joblib==0.17.0 scikit-learn==0.23.2 pandas==1.1.4 matplotlib==3.3.2",
        "pip install tqdm biopython Levenshtein",
        "pip install modlamp==4.2.3 --no-deps",
        "pip install git+https://github.com/szczurek-lab/hydramp.git@6590d2f4c2963f25d30669052a4c4a857e0e7279 --no-deps",
    ],
)

models_volume = Volume(name="amp-models", mount_path="/models")


@function(
    gpu=["RTX4090", "A10G"],
    image=image,
    memory="24Gi",
    cpu=8,
    volumes=[models_volume],
    timeout=1800,
)
def run_scaled_hydramp_generation(
    seed_records: list,
    n_unconstrained: int = 2000,
    n_analogue_per_seed: int = 35,  # 60 seeds * 35 attempts * 2 temps = 4,200 attempts
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
    model_path = "hydramp_checkpoint/model"
    decomposer_path = "hydramp_checkpoint/pca_decomposer.joblib"

    print(f"[+] Loading HydrAMPGenerator from {model_path}...")
    t0 = time.time()
    generator = HydrAMPGenerator(
        model_path=model_path,
        decomposer_path=decomposer_path,
        softmax=True,
    )
    print(f"✓ Generator loaded in {time.time() - t0:.2f}s")

    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

    # 1. Mode 1: 2,000 Unconstrained Candidates
    print(f"\n--- Generating {n_unconstrained} Mode 1 (Unconstrained) Candidates ---")
    t1 = time.time()
    unconstrained_raw = generator.unconstrained_generation(
        mode="amp",
        n_target=n_unconstrained,
        seed=base_seed,
        filter_out=True,
        properties=True,
        n_attempts=1,
    )
    dt1 = time.time() - t1
    print(f"✓ Mode 1 generated {len(unconstrained_raw)} sequences in {dt1:.2f}s")

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
            "seed": base_seed + i,
            "model": "HydrAMP_epoch37",
        })

    # 2. Mode 2: Analogue Perturbation across 60 Seeds at Calibrated Temps (2.0 and 3.0)
    calibrated_temps = [2.0, 3.0]
    print(f"\n--- Generating Mode 2 (Analogues) across {len(seed_records)} Seeds at Temps {calibrated_temps} ---")
    t2 = time.time()

    analogue_records = []
    seed_seq_to_meta = {r["sequence"]: r for r in seed_records}
    seeds_list = [r["sequence"] for r in seed_records]

    for temp_val in calibrated_temps:
        t_temp = time.time()
        print(f"[*] Running Analogue Perturbation at Temp={temp_val} (attempts/seed={n_analogue_per_seed})...")

        analogue_batch = generator.analogue_generation(
            sequences=seeds_list,
            seed=base_seed + int(temp_val * 1000),
            filtering_criteria="discovery",
            n_attempts=n_analogue_per_seed,
            temp=temp_val,
        )

        for parent_seq, p_data in analogue_batch.items():
            meta = seed_seq_to_meta.get(parent_seq, {})
            children = p_data.get("generated_sequences", [])
            if children:
                for c in children:
                    c_seq = str(c["sequence"]).strip().upper()
                    if c_seq == parent_seq:
                        continue  # discard exact parent echo

                    is_canonical = set(c_seq).issubset(STANDARD_AAS)
                    len_ok = 8 <= len(c_seq) <= 25
                    is_valid = is_canonical and len_ok and len(c_seq) > 0
                    reason = "ok" if is_valid else ("invalid_residue" if not is_canonical else "invalid_length")

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
                        "seed": base_seed,
                        "model": "HydrAMP_epoch37",
                    })

        dt_temp = time.time() - t_temp
        print(f"  Temp={temp_val}: completed in {dt_temp:.2f}s")

    dt2 = time.time() - t2
    print(f"✓ Mode 2 generated {len(analogue_records)} analogue candidates in {dt2:.2f}s")

    return {
        "status": "success",
        "unconstrained_records": unconstrained_records,
        "analogue_records": analogue_records,
        "unconstrained_time_sec": round(dt1, 2),
        "analogue_time_sec": round(dt2, 2),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    proto_csv = root / "vae-latent-models/outputs/prototype_panel_validated.csv"
    proto_df = pd.read_csv(proto_csv)
    eligible_seeds = proto_df[proto_df["hydramp_eligible"]].to_dict("records")
    print(f"Loaded {len(eligible_seeds)} eligible seeds for scaled generation.")

    N_UNCONSTRAINED = 2000
    N_ATTEMPTS_PER_SEED = 35

    print("=======================================================")
    print(" Running Scaled VAE Generation on Beam Cloud RTX 4090")
    print(f" Target Mode 1: {N_UNCONSTRAINED} Unconstrained Candidates")
    print(f" Target Mode 2: {len(eligible_seeds)} Seeds x 2 Temps (~2,000 Analogue Candidates)")
    print("=======================================================\n")

    t0 = time.time()
    res = run_scaled_hydramp_generation.remote(
        seed_records=eligible_seeds,
        n_unconstrained=N_UNCONSTRAINED,
        n_analogue_per_seed=N_ATTEMPTS_PER_SEED,
        base_seed=42,
    )
    total_time = time.time() - t0
    print(f"\n✓ Scaled generation finished in {total_time:.1f}s!")

    df1 = pd.DataFrame(res["unconstrained_records"])
    df2 = pd.DataFrame(res["analogue_records"])
    full_df = pd.concat([df1, df2], ignore_index=True)

    out_raw = root / "vae-latent-models/outputs/scaled_raw_candidates.csv"
    full_df.to_csv(out_raw, index=False)
    print(f"Saved {len(full_df)} scaled raw candidates ({len(df1)} Mode 1 + {len(df2)} Mode 2) to {out_raw}")


if __name__ == "__main__":
    main()
