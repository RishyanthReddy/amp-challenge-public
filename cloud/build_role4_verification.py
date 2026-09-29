"""Build a current Role 04 record from the latest generation and audit manifests."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data-engineering/src"))
from amp_data.core import fasta_rows, norm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw_path = ROOT / "diffusion-models/outputs/scaled_raw_diffusion_candidates.csv"
    candidate_path = ROOT / "outputs/diffusion_candidates.csv"
    generation_manifest_path = ROOT / "diffusion-models/reports/diffusion_generation_manifest.json"
    audit_manifest_path = ROOT / "diffusion-models/reports/diffusion_candidate_audit_manifest.json"
    checkpoint_path = ROOT / "cloud/diffusion_checkpoint/model.pt"
    reference_path = ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    for path in (raw_path, candidate_path, generation_manifest_path, audit_manifest_path, checkpoint_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Role 04 artifact is missing: {path}")

    raw = pd.read_csv(raw_path)
    candidates = pd.read_csv(candidate_path)
    generation = json.loads(generation_manifest_path.read_text())
    audit = json.loads(audit_manifest_path.read_text())
    pipeline_snapshot_path = ROOT / generation["pipeline_source_snapshot_file"]
    if not pipeline_snapshot_path.is_file():
        raise FileNotFoundError(f"Role 04 executed pipeline source snapshot is missing: {pipeline_snapshot_path}")
    pipeline_snapshot_sha = sha256_file(pipeline_snapshot_path)
    if (
        pipeline_snapshot_sha != generation.get("pipeline_source_sha256")
        or pipeline_snapshot_sha != generation.get("pipeline_source_snapshot_sha256")
        or audit.get("pipeline_source_sha256") != pipeline_snapshot_sha
        or audit.get("pipeline_source_snapshot_sha256") != pipeline_snapshot_sha
    ):
        raise ValueError("Role 04 executed pipeline source hash does not match its generation/audit manifests")
    raw_sha = sha256_file(raw_path)
    candidate_sha = sha256_file(candidate_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    for name, manifest, key, actual in (
        ("generation raw output", generation, "candidate_csv_sha256", raw_sha),
        ("audited candidates", audit, "candidate_csv_sha256", candidate_sha),
        ("audit raw input", audit, "raw_candidate_csv_sha256", raw_sha),
        ("local checkpoint", generation, "checkpoint_sha256", checkpoint_sha),
        ("audit checkpoint", audit, "checkpoint_sha256", checkpoint_sha),
    ):
        if manifest.get(key) != actual:
            raise ValueError(f"Role 04 {name} hash mismatch for {key}")

    refs, _, _ = fasta_rows(reference_path)
    reference_set = {norm(row["sequence"]) for row in refs}
    exact_matches = int(candidates["sequence"].map(lambda sequence: norm(sequence) in reference_set).sum())
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    valid_mask = candidates["sequence"].map(
        lambda sequence: 8 <= len(str(sequence)) <= 50 and set(str(sequence)) <= standard
    )
    if not valid_mask.all():
        raise ValueError(f"Role 04 candidate export has {(~valid_mask).sum()} invalid sequences")
    if exact_matches:
        raise ValueError(f"Role 04 candidate export has {exact_matches} exact reference matches")
    if len(raw) != int(generation["production_rows"]):
        raise ValueError("Role 04 raw row count disagrees with generation manifest")
    if len(candidates) != int(audit["retained_candidate_rows"]):
        raise ValueError("Role 04 candidate count disagrees with audit manifest")

    provenance_columns = (
        "checkpoint_sha256", "esm_weights_sha256", "esm_regression_sha256",
        "model_source_sha256", "torch_version", "numpy_version", "pandas_version",
        "fair_esm_version", "ema_pytorch_version", "einops_version",
    )
    for column in provenance_columns:
        if column not in raw or raw[column].nunique(dropna=True) != 1:
            raise ValueError(f"Role 04 run must have one recorded value for {column}")

    verification = {
        "role": "04_sequence_diffusion",
        "run_id": generation["run_id"],
        "gpu": generation["gpu"],
        "production_rows": int(len(raw)),
        "retained_candidate_rows": int(len(candidates)),
        "unique_retained_sequences": int(candidates["sequence"].nunique()),
        "exact_reference_matches": exact_matches,
        "reference_sequence_count": int(len(reference_set)),
        "all_candidates_canonical_and_length_valid": bool(valid_mask.all()),
        "retained_novel_le80_count": int((candidates["max_reference_similarity"] <= 0.80).sum()),
        "retained_synthesizable_count": int(candidates["synthesizable"].sum()),
        "checkpoint_sha256": checkpoint_sha,
        "esm_weights_sha256": generation["esm_weights_sha256"],
        "esm_regression_sha256": generation["esm_regression_sha256"],
        "model_source_sha256": generation["model_source_sha256"],
        "executed_pipeline_source_sha256": pipeline_snapshot_sha,
        "current_pipeline_source_sha256": sha256_file(ROOT / "cloud/run_diffusion_pipeline.py"),
        "candidate_csv_sha256": candidate_sha,
        "raw_candidate_csv_sha256": raw_sha,
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "audit_manifest_sha256": sha256_file(audit_manifest_path),
        "runtime": generation["runtime"],
        "sampler": generation["sampler"],
        "feasibility_metrics": generation["feasibility_metrics"],
        "step_comparison_results": generation["step_comparison_results"],
    }
    out_path = ROOT / "diffusion-models/reports/artifact_verification.json"
    out_path.write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Role 04 current artifacts verified; {len(candidates)} retained candidates, 0 exact overlaps.")
    print(f"Candidate SHA-256: {candidate_sha}")


if __name__ == "__main__":
    main()
