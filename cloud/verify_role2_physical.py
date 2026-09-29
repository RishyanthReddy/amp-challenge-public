"""Recompute current Role 02 candidate, novelty, and generation-provenance checks."""

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data-engineering/src"))
from amp_data.core import fasta_rows, lev_ratio, norm

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw_path = ROOT / "autoregressive-models/outputs/generated_candidates_all.csv"
    candidate_path = ROOT / "outputs/ar_candidates.csv"
    role_candidate_path = ROOT / "autoregressive-models/outputs/finetuned_candidates.csv"
    metrics_path = ROOT / "autoregressive-models/outputs/finetuned_metrics.json"
    manifest_path = ROOT / "autoregressive-models/outputs/ar_candidate_audit_manifest.json"
    reference_path = ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    for path in (raw_path, candidate_path, role_candidate_path, metrics_path, manifest_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Role 02 artifact is missing: {path}")

    raw = pd.read_csv(raw_path)
    candidates = pd.read_csv(candidate_path)
    metrics = json.loads(metrics_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    raw_sha = sha256_file(raw_path)
    candidate_sha = sha256_file(candidate_path)
    if role_candidate_path.read_bytes() != candidate_path.read_bytes():
        raise ValueError("Role 02 role-level and root candidate exports differ")
    if manifest.get("raw_candidate_source_sha256") != raw_sha:
        raise ValueError("Role 02 raw generation source hash disagrees with its audit manifest")
    if manifest.get("candidate_csv_sha256") != candidate_sha:
        raise ValueError("Role 02 candidate output hash disagrees with its audit manifest")
    if manifest.get("generation_metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("Role 02 generation metrics hash disagrees with its audit manifest")
    if metrics.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise ValueError("Role 02 checkpoint hashes disagree across run and audit records")
    if len(raw) != int(manifest["raw_generated_rows"]) or len(candidates) != int(manifest["candidate_rows"]):
        raise ValueError("Role 02 candidate row counts disagree with the audit manifest")

    reference_rows, _, _ = fasta_rows(reference_path)
    references = {norm(row["sequence"]) for row in reference_rows}
    if len(references) != 39_448:
        raise ValueError(f"Expected 39,448 unique references, found {len(references)}")
    refs_by_length: dict[int, list[str]] = {}
    for reference in references:
        refs_by_length.setdefault(len(reference), []).append(reference)

    exact_matches = 0
    novelty_eligible = 0
    recomputed_max = 0.0
    for row in candidates.itertuples(index=False):
        sequence = str(row.sequence).strip().upper()
        if not (8 <= len(sequence) <= 50) or not set(sequence) <= STANDARD_AA:
            raise ValueError(f"Role 02 candidate has invalid length/alphabet: {sequence!r}")
        exact = sequence in references
        exact_matches += int(exact)
        lo, hi = math.ceil(len(sequence) * 2 / 3), math.floor(len(sequence) * 1.5)
        maximum = max(
            (lev_ratio(sequence, reference)
             for length in range(lo, hi + 1)
             for reference in refs_by_length.get(length, ())),
            default=0.0,
        )
        if not math.isclose(maximum, float(row.max_reference_similarity), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Role 02 novelty score mismatch for candidate {sequence}")
        expected_flag = maximum <= 0.80
        if bool(row.passes_novelty_rule_le80) != expected_flag:
            raise ValueError(f"Role 02 novelty flag mismatch for candidate {sequence}")
        novelty_eligible += int(expected_flag)
        recomputed_max = max(recomputed_max, maximum)

    if exact_matches:
        raise ValueError(f"Role 02 retained {exact_matches} exact reference matches")
    if novelty_eligible != int(manifest["novel_le80_count"]):
        raise ValueError("Role 02 novelty-eligible count disagrees with its audit manifest")

    verification = {
        "role": "02_autoregressive_generation",
        "raw_generated_rows": int(len(raw)),
        "retained_candidate_rows": int(len(candidates)),
        "unique_candidate_sequences": int(candidates["sequence"].nunique()),
        "exact_reference_matches_quarantined": int(manifest["exact_reference_matches_quarantined"]),
        "exact_reference_matches_retained": exact_matches,
        "novelty_eligible_count": novelty_eligible,
        "recomputed_max_reference_similarity": recomputed_max,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "raw_candidate_source_sha256": raw_sha,
        "candidate_csv_sha256": candidate_sha,
        "generation_metrics_sha256": manifest["generation_metrics_sha256"],
        "model_code_revision": manifest["model_code_revision"],
        "reference_sequence_count": len(references),
    }
    out_path = ROOT / "autoregressive-models/outputs/artifact_verification.json"
    out_path.write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Role 02 current artifacts verified; {len(candidates)} retained candidates, 0 exact references.")
    print(f"Candidate SHA-256: {candidate_sha}")


if __name__ == "__main__":
    main()
