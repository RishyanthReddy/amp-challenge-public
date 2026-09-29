"""Build a current Role 05 artifact record and verify the exported ancestry DAG."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evolutionary-search/src"))
sys.path.insert(0, str(ROOT / "data-engineering/src"))

from evolution.ga import validate_ancestry
from amp_data.core import fasta_rows, norm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    out_dir = ROOT / "outputs"
    reports_dir = ROOT / "evolutionary-search/reports"
    candidate_path = out_dir / "evolution_candidates.csv"
    nodes_path = out_dir / "evolution_candidate_nodes.csv"
    ancestry_path = out_dir / "ancestry.csv"
    seeds_path = ROOT / "evolutionary-search/data/evolution_seeds.csv"
    run_manifest_path = reports_dir / "production_run_manifest.json"
    reference_path = ROOT / "data-engineering/data/challenge/antibacterial.fasta"

    for path in (candidate_path, nodes_path, ancestry_path, seeds_path, run_manifest_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Role 05 artifact is missing: {path}")

    df_candidates = pd.read_csv(candidate_path)
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(ancestry_path)
    df_seeds = pd.read_csv(seeds_path)
    run_manifest = json.loads(run_manifest_path.read_text())

    node_records = df_nodes.to_dict(orient="records")
    for node in node_records:
        if pd.isna(node.get("parent_id")):
            node["parent_id"] = None
    validate_ancestry(node_records, df_edges.to_dict(orient="records"))
    if df_candidates["sequence"].duplicated().any():
        raise ValueError("Exported evolution candidate pool contains duplicate sequences")
    if not set(df_candidates["sequence"]).issubset(set(df_nodes["sequence"])):
        raise ValueError("Deduplicated candidates contain sequences absent from the node table")

    standard = set("ACDEFGHIKLMNPQRSTVWY")
    if not df_nodes["sequence"].map(lambda s: 8 <= len(str(s)) <= 50 and set(str(s)) <= standard).all():
        raise ValueError("Evolutionary node table contains invalid amino-acid sequences")
    refs, _, _ = fasta_rows(reference_path)
    reference_set = {norm(row["sequence"]) for row in refs}
    exact_reference_matches = int(
        df_candidates["sequence"].map(lambda sequence: norm(sequence) in reference_set).sum()
    )
    if exact_reference_matches:
        raise ValueError(
            f"Final evolutionary candidate pool contains {exact_reference_matches} exact reference matches"
        )

    actual_hashes = {
        "output_candidate_sha256": sha256_file(candidate_path),
        "output_candidate_nodes_sha256": sha256_file(nodes_path),
        "output_ancestry_sha256": sha256_file(ancestry_path),
        "seed_file_sha256": sha256_file(seeds_path),
    }
    for key, value in actual_hashes.items():
        if run_manifest.get(key) != value:
            raise ValueError(f"Run manifest hash mismatch for {key}")

    children = set(df_nodes.loc[df_nodes["parent_id"].notna(), "sequence_id"])
    edge_children = set(df_edges["child_id"])
    verification = {
        "role": "05_evolutionary_search",
        "algorithm": run_manifest["algorithm"],
        "run_id": run_manifest["run_id"],
        "seed": 42,
        "seed_count": int(len(df_seeds)),
        "family_count": int(df_seeds["family_id"].nunique()),
        "candidate_node_count": int(len(df_nodes)),
        "unique_candidate_count": int(df_candidates["sequence"].nunique()),
        "ancestry_edges": int(len(df_edges)),
        "all_nonroot_nodes_have_one_valid_parent_edge": edge_children == children,
        "lineage_validation": "PASS",
        "exact_reference_matches": exact_reference_matches,
        "all_canonical_and_length_valid": True,
        "candidate_sha256": actual_hashes["output_candidate_sha256"],
        "candidate_nodes_sha256": actual_hashes["output_candidate_nodes_sha256"],
        "ancestry_sha256": actual_hashes["output_ancestry_sha256"],
        "seed_file_sha256": actual_hashes["seed_file_sha256"],
        "evaluator_ledger": run_manifest["evaluator_ledger"],
        "termination_reason": run_manifest["termination_reason"],
    }
    verification_path = reports_dir / "artifact_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    print(f"✓ Current candidate/node/ancestry artifacts validated; report: {verification_path}")


if __name__ == "__main__":
    main()
