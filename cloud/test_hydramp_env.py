"""
Test HydrAMP Environment & Model Load Smoke Test on Beam Cloud
Verifies:
1. Python 3.8 + TensorFlow 2.2.1 + Keras 2.3.1 on Linux container
2. HydrAMP package installation and import
3. Checkpoint and PCA decomposer loading
4. Small test generation in both modes (unconstrained + analogue)
"""

import os
import sys
import time
import json
from pathlib import Path

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
    memory="16Gi",
    cpu=4,
    volumes=[models_volume],
    timeout=600,
)
def smoke_test_hydramp(test_seeds: list) -> dict:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    import keras
    import joblib
    import numpy as np
    from amp.inference.inference import HydrAMPGenerator

    print(f"🚀 Worker Python: {sys.version}")
    print(f"📦 TensorFlow Version: {tf.__version__}")
    print(f"📦 Keras Version: {keras.__version__}")
    print(f"🎮 GPU Available: {tf.test.is_gpu_available()}")

    # Path to checkpoint synced in cloud directory
    model_path = "hydramp_checkpoint/model"
    decomposer_path = "hydramp_checkpoint/pca_decomposer.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found in container!")
    if not os.path.exists(decomposer_path):
        raise FileNotFoundError(f"Decomposer path {decomposer_path} not found in container!")

    print(f"[+] Loading HydrAMPGenerator from {model_path}...")
    t0 = time.time()
    generator = HydrAMPGenerator(
        model_path=model_path,
        decomposer_path=decomposer_path,
        softmax=True,
    )
    load_time = time.time() - t0
    print(f"✓ HydrAMPGenerator loaded successfully in {load_time:.2f}s!")

    # Test Mode 1: Unconstrained generation (5 samples)
    print("\n--- Testing Mode 1: Unconstrained Generation (5 samples) ---")
    t_unconstrained = time.time()
    unconstrained_res = generator.unconstrained_generation(
        mode="amp",
        n_target=5,
        seed=42,
        filter_out=True,
        properties=True,
        n_attempts=1,
    )
    dt_unconstrained = time.time() - t_unconstrained
    print(f"✓ Mode 1 generated {len(unconstrained_res)} sequences in {dt_unconstrained:.2f}s")
    for r in unconstrained_res:
        print(f"  Seq: {r['sequence']} | AMP prob: {r.get('amp', 0):.4f} | MIC prob: {r.get('mic', 0):.4f}")

    # Test Mode 2: Analogue generation (from test seeds)
    print("\n--- Testing Mode 2: Analogue Generation (3 seeds) ---")
    t_analogue = time.time()
    analogue_dict = generator.analogue_generation(
        sequences=test_seeds[:3],
        seed=42,
        filtering_criteria="discovery",
        n_attempts=20,
        temp=2.0,
    )
    dt_analogue = time.time() - t_analogue
    print(f"✓ Mode 2 generated analogues across {len(analogue_dict)} parent seeds in {dt_analogue:.2f}s")
    
    analogue_samples = []
    for parent_seq, p_data in analogue_dict.items():
        children = p_data.get("generated_sequences", [])
        if children:
            for c in children[:2]:
                analogue_samples.append({
                    "parent": parent_seq,
                    "child": c["sequence"],
                    "amp": c.get("amp", 0),
                    "mic": c.get("mic", 0),
                })
                print(f"  Parent: {parent_seq} -> Child: {c['sequence']} (AMP: {c.get('amp', 0):.3f})")

    return {
        "status": "success",
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "gpu_available": tf.test.is_gpu_available(),
        "load_time_sec": round(load_time, 2),
        "unconstrained_samples": unconstrained_res,
        "analogue_samples": analogue_samples,
    }

def main():
    root = Path(__file__).resolve().parents[1]
    test_seeds = [
        "AKRKLVWQ",
        "GLLDFVTGVGKGIFAAL",
        "KWKLFKKIPKFLHLAKKF",
    ]
    print("Dispatching HydrAMP Environment & Model Load Smoke Test to Beam Cloud...")
    res = smoke_test_hydramp.remote(test_seeds=test_seeds)
    print("\n================== SMOKE TEST RESULT ==================")
    print("Status:", res["status"])
    print("TensorFlow:", res["tf_version"])
    print("Keras:", res["keras_version"])
    print("GPU Available:", res["gpu_available"])
    print("Model Load Time:", res["load_time_sec"], "seconds")
    print(f"Unconstrained Samples ({len(res['unconstrained_samples'])}):")
    for s in res["unconstrained_samples"]:
        print(f"  - {s['sequence']} (AMP: {s.get('amp', 0):.3f}, MIC: {s.get('mic', 0):.3f})")
    print(f"Analogue Samples ({len(res['analogue_samples'])}):")
    for s in res["analogue_samples"]:
        print(f"  - Parent: {s['parent']} -> Child: {s['child']}")
    print("=======================================================")

    log_path = root / "vae-latent-models/outputs/model_load.log"
    with open(log_path, "w") as f:
        f.write(f"HydrAMP Model Load Smoke Test on Beam Cloud\n")
        f.write(f"Status: {res['status']}\n")
        f.write(f"TensorFlow: {res['tf_version']}, Keras: {res['keras_version']}\n")
        f.write(f"GPU Available: {res['gpu_available']}\n")
        f.write(f"Load Time: {res['load_time_sec']}s\n")
        f.write(f"Unconstrained Samples: {len(res['unconstrained_samples'])}\n")
        f.write(f"Analogue Samples: {len(res['analogue_samples'])}\n")
    print(f"✓ Saved smoke test log to {log_path}")

if __name__ == "__main__":
    main()
