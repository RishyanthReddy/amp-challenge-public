"""Recompute current Role 03 generation, checkpoint, ancestry, and sequence checks."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw_path = ROOT / "vae-latent-models/outputs/raw_generated_pool.csv"
    candidate_path = ROOT / "outputs/vae_candidates.csv"
    role_candidate_path = ROOT / "vae-latent-models/outputs/vae_candidates.csv"
    manifest_path = ROOT / "vae-latent-models/outputs/hydramp_generation_manifest.json"
    checkpoint_path = ROOT / "vae-latent-models/checkpoint/model"
    pca_path = ROOT / "vae-latent-models/checkpoint/pca_decomposer.joblib"
    reference_path = ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    prototype_path = ROOT / "vae-latent-models/outputs/prototype_panel_validated.csv"
    for path in (raw_path, candidate_path, role_candidate_path, manifest_path, checkpoint_path, pca_path, reference_path, prototype_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Role 03 artifact is missing: {path}")

    raw = pd.read_csv(raw_path)
    candidates = pd.read_csv(candidate_path)
    manifest = json.loads(manifest_path.read_text())
    if role_candidate_path.read_bytes() != candidate_path.read_bytes():
        raise ValueError("Role 03 candidate exports differ")
    actual_hashes = {
        "raw_candidate_csv_sha256": sha256_file(raw_path),
        "candidate_csv_sha256": sha256_file(candidate_path),
        "checkpoint_tree_sha256": sha256_tree(checkpoint_path),
        "pca_decomposer_sha256": sha256_file(pca_path),
    }
    for key, value in actual_hashes.items():
        if manifest.get(key) != value:
            raise ValueError(f"Role 03 manifest hash mismatch for {key}")
    if len(raw) != int(manifest["raw_candidate_count"]):
        raise ValueError("Role 03 raw row count disagrees with its manifest")
    if len(candidates) != int(manifest["post_audit_candidate_rows"]):
        raise ValueError("Role 03 audited row count disagrees with its manifest")

    refs = set()
    for block in reference_path.read_text().split(">")[1:]:
        rows = block.splitlines()
        refs.add("".join("".join(rows[1:]).split()).upper())
    sequences = candidates["sequence"].astype(str).str.strip().str.upper()
    valid = sequences.map(lambda sequence: 8 <= len(sequence) <= 50 and set(sequence) <= STANDARD)
    exact_matches = int(sequences.isin(refs).sum())
    if not valid.all() or exact_matches:
        raise ValueError("Role 03 candidates violate canonical length/alphabet or exact-reference checks")

    prototype = pd.read_csv(prototype_path)
    eligible_seed_ids = set(prototype.loc[prototype["hydramp_eligible"].astype(bool), "seed_id"].astype(str))
    analogues = candidates[candidates["generation_mode"] == "analogue"]
    parents = set(analogues["parent_sequence_id"].dropna().astype(str))
    if analogues["parent_sequence_id"].isna().any() or not parents.issubset(eligible_seed_ids):
        raise ValueError("Role 03 analogue ancestry does not resolve to eligible prototypes")

    verification = {
        "role": "03_hydramp_generation",
        "upstream_revision": manifest["upstream_revision"],
        "device": manifest["device"],
        "raw_rows": int(len(raw)),
        "audited_candidate_rows": int(len(candidates)),
        "unique_candidate_sequences": int(sequences.nunique()),
        "exact_reference_matches": exact_matches,
        "reference_sequence_count": int(len(refs)),
        "analogue_rows": int(len(analogues)),
        "analogue_parent_count": int(len(parents)),
        "all_analogue_parents_are_eligible": True,
        "novel_le80_count": int((candidates["max_reference_similarity"] <= 0.80).sum()),
        "synthesizable_count": int(candidates["synthesizable"].sum()),
        **actual_hashes,
        "runtime": {key: manifest[key] for key in ("tensorflow_version", "keras_version")},
    }
    out_path = ROOT / "vae-latent-models/outputs/artifact_verification.json"
    out_path.write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Role 03 current artifacts verified; {len(candidates)} audited rows, 0 exact overlaps.")
    print(f"Candidate SHA-256: {actual_hashes['candidate_csv_sha256']}")


if __name__ == "__main__":
    main()
