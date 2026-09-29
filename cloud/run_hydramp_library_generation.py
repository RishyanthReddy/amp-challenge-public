"""Generate a reproducible 50,000-sequence HydrAMP auxiliary library on Beam.

Run this script from a small Beam bundle containing the adjacent ``generate.py``
module and ``data/antibacterial.fasta``. The checkpoint is read from the existing
``amp-models`` volume, so no model weights or unrelated workspace files are synced.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from beam import Image, Volume, function


HYDRAMP_UPSTREAM_REVISION = "6590d2f4c2963f25d30669052a4c4a857e0e7279"
N_LIBRARY = 50_000
BASE_SEED = 42

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@function(
    image=image,
    memory="24Gi",
    cpu=8,
    volumes=[models_volume],
    timeout=2400,
)
def generate_full_library(n_sequences: int = N_LIBRARY, seed: int = BASE_SEED) -> dict:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    import keras
    from importlib.metadata import version as package_version
    from amp.inference.inference import HydrAMPGenerator
    from generate import _read_fasta_sequences, generate_library

    bundle_dir = Path(__file__).resolve().parent
    reference_path = bundle_dir / "data/antibacterial.fasta"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference FASTA is missing from Beam bundle: {reference_path}")
    model_path = Path("/models/hydramp/model")
    decomposer_path = Path("/models/hydramp/pca_decomposer.joblib")
    if not model_path.is_dir() or not decomposer_path.is_file():
        raise FileNotFoundError("HydrAMP weights or PCA decomposer are missing from Beam volume")

    def tree_hash(path: Path) -> str:
        digest = hashlib.sha256()
        for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    references = set(_read_fasta_sequences(reference_path))
    generator = HydrAMPGenerator(
        model_path=str(model_path),
        decomposer_path=str(decomposer_path),
        softmax=True,
    )
    rounds = []
    original_generate = generator.unconstrained_generation

    def counted_generate(*args, **kwargs):
        records = original_generate(*args, **kwargs)
        rounds.append({
            "seed": kwargs.get("seed"),
            "requested": kwargs.get("n_target"),
            "returned": len(records),
        })
        return records

    generator.unconstrained_generation = counted_generate
    start = time.time()
    library = generate_library(generator, n_sequences, seed, references)
    sequences = list(library)
    return {
        "status": "success",
        "sequences": sequences,
        "requested_count": n_sequences,
        "actual_count": len(sequences),
        "unique_count": len(set(sequences)),
        "exact_reference_overlap_count": len(set(sequences) & references),
        "reference_count": len(references),
        "base_seed": seed,
        "generation_rounds": rounds,
        "checkpoint_tree_sha256": tree_hash(model_path),
        "pca_decomposer_sha256": sha256_file(decomposer_path),
        "upstream_revision": HYDRAMP_UPSTREAM_REVISION,
        "tensorflow_version": str(tf.__version__),
        "keras_version": str(keras.__version__),
        "numpy_version": package_version("numpy"),
        "device": tf.test.gpu_device_name() or "CPU",
        "elapsed_seconds": round(time.time() - start, 2),
    }


def main() -> None:
    bundle_dir = Path(__file__).resolve().parent
    project_root = bundle_dir.parent
    reference_path = bundle_dir / "data/antibacterial.fasta"
    reference_sequences = set(
        "".join("".join(block.splitlines()[1:]).split()).upper()
        for block in reference_path.read_text().split(">")
        if block.strip()
    )

    result = generate_full_library.remote(n_sequences=N_LIBRARY, seed=BASE_SEED)
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError("Beam did not return a successful HydrAMP library run")
    sequences = [str(sequence).strip().upper() for sequence in result["sequences"]]
    if len(sequences) != N_LIBRARY or len(set(sequences)) != N_LIBRARY:
        raise ValueError(f"Expected {N_LIBRARY} unique sequences; got {len(sequences)} records / {len(set(sequences))} unique")
    if set(sequences) & reference_sequences:
        raise ValueError("HydrAMP full-library output contains exact challenge-reference matches")
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    if any(not (8 <= len(sequence) <= 50) or not set(sequence) <= standard for sequence in sequences):
        raise ValueError("HydrAMP full-library output contains an invalid sequence")

    out_dir = project_root / "vae-latent-models/outputs/intermediate_hydramp_fastas"
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "library.fasta"
    with fasta_path.open("w") as handle:
        for index, sequence in enumerate(sequences, start=1):
            handle.write(f">hydra_lib_{index:05d}\n{sequence}\n")

    manifest = {key: value for key, value in result.items() if key != "sequences"}
    manifest.update({
        "role": "03_hydramp_auxiliary_library_generation",
        "generation_mode": "unconstrained_default_hydramp_amp",
        "filter_out": True,
        "softmax": True,
        "generator_source_sha256": sha256_file(bundle_dir / "generate.py"),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "reference_fasta_sha256": sha256_file(reference_path),
        "library_fasta_sha256": sha256_file(fasta_path),
    })
    manifest_path = out_dir / "library_generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"HydrAMP library generation verified: {len(sequences)} unique candidates, 0 exact overlaps.")
    print(f"Library SHA-256: {manifest['library_fasta_sha256']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
