"""Recompute Role 06 master-score, model, and full APEX coverage checks."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data-engineering/src"))
sys.path.insert(0, str(ROOT / "shared-evaluator/src"))
from amp_data.core import fasta_rows, norm
from evaluator.apex_scorer import PATHOGEN_PANEL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    master_path = ROOT / "outputs/master_scored_candidates.csv"
    manifest_path = ROOT / "shared-evaluator/reports/master_evaluation_manifest.json"
    reference_path = ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    source_paths = {
        "autoregressive": ROOT / "outputs/ar_candidates.csv",
        "vae_latent": ROOT / "outputs/vae_candidates.csv",
        "diffusion": ROOT / "outputs/diffusion_candidates.csv",
        "evolution": ROOT / "outputs/evolution_candidates.csv",
    }
    for path in (master_path, manifest_path, reference_path, *source_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Required Role 06 artifact is missing: {path}")

    master = pd.read_csv(master_path)
    manifest = json.loads(manifest_path.read_text())
    actual_source_hashes = {domain: sha256_file(path) for domain, path in source_paths.items()}
    if manifest.get("input_sha256") != actual_source_hashes:
        raise ValueError("Role 06 source candidate hashes disagree with its evaluation manifest")
    if manifest.get("output_sha256") != sha256_file(master_path):
        raise ValueError("Role 06 scored master hash disagrees with its evaluation manifest")
    if len(master) != int(manifest["unique_sequences"]) or master["sequence"].duplicated().any():
        raise ValueError("Role 06 master row count or sequence uniqueness disagrees with its manifest")

    refs, _, _ = fasta_rows(reference_path)
    reference_set = {norm(row["sequence"]) for row in refs}
    exact_matches = int(master["sequence"].map(lambda sequence: norm(sequence) in reference_set).sum())
    if exact_matches:
        raise ValueError(f"Role 06 master contains {exact_matches} exact reference matches")

    valid = master["is_valid"].fillna(False).astype(bool)
    synthesizable = master["synthesizable"].fillna(False).astype(bool)
    novelty_flag = master["passes_novelty_rule_le80"].fillna(False).astype(bool)
    similarity = pd.to_numeric(master["max_reference_similarity"], errors="coerce")
    if not np.isfinite(similarity.to_numpy(dtype=float)).all():
        raise ValueError("Role 06 master contains non-finite reference similarity values")
    similarity_rule = similarity <= 0.80
    if not np.array_equal(novelty_flag.to_numpy(dtype=bool), similarity_rule.to_numpy(dtype=bool)):
        raise ValueError("Role 06 novelty flags disagree with max_reference_similarity <= 0.80")
    rankable = valid & synthesizable & novelty_flag & similarity_rule
    apex_missing = master.loc[rankable, PATHOGEN_PANEL].isna().any(axis=1)
    if apex_missing.any():
        raise ValueError(f"Role 06 has {int(apex_missing.sum())} rankable sequences missing APEX panel values")
    if int(rankable.sum()) != int(manifest["rankable_sequences_scored_by_apex"]):
        raise ValueError("Role 06 rankable count disagrees with its evaluation manifest")
    if not set(master["primary_domain"].dropna()).issubset(set(source_paths)):
        raise ValueError("Role 06 master contains an unknown primary generation domain")
    if set(source_paths) - set(";".join(master["contributing_domains"].dropna().astype(str)).split(";")):
        raise ValueError("Role 06 master does not contain all four current source domains")
    if not np.isfinite(master.loc[valid, "pred_amp_probability"].to_numpy(dtype=float)).all():
        raise ValueError("Role 06 valid candidate AMP predictions contain non-finite values")
    if not np.isfinite(master.loc[valid, "pred_empirical_hemolysis_prob"].to_numpy(dtype=float)).all():
        raise ValueError("Role 06 valid empirical hemolysis predictions contain non-finite values")

    verification = {
        "role": "06_shared_evaluator",
        "master_rows": int(len(master)),
        "unique_sequences": int(master["sequence"].nunique()),
        "valid_sequences": int(valid.sum()),
        "rankable_sequences": int(rankable.sum()),
        "novelty_eligible_sequences": int(novelty_flag.sum()),
        "apex_rows_complete_for_rankable": int(rankable.sum()),
        "apex_pathogen_count": int(len(PATHOGEN_PANEL)),
        "exact_reference_matches": exact_matches,
        "primary_domains": sorted(set(master["primary_domain"].dropna().astype(str))),
        "input_sha256": actual_source_hashes,
        "master_sha256": sha256_file(master_path),
        "classifier_model_sha256": manifest["classifier_model_sha256"],
        "apex_asset_sha256": manifest["apex_asset_sha256"],
        "evaluator_code_sha256": manifest["evaluator_code_sha256"],
        "runtime": manifest["runtime"],
    }
    out_path = ROOT / "shared-evaluator/reports/artifact_verification.json"
    out_path.write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Role 06 current artifacts verified; {len(master)} unique sequences, full APEX coverage for {int(rankable.sum())} rankable candidates.")
    print(f"Master SHA-256: {verification['master_sha256']}")


if __name__ == "__main__":
    main()
