"""Select an isolated AR/evolution challenger without changing the research portfolio.

The input is the hash-pinned Role 06 master table. Sequences with any HydrAMP or
AMP-Diffusion contribution are excluded, including cross-domain duplicates. Only
valid, synthesizable, novelty-eligible, APEX-scored rows may be ranked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "portfolio-selection/src"))

from portfolio.dpp_optimizer import DppPortfolioOptimizer
from portfolio.map_elites import MapElitesArchive

ENTRY_DOMAINS = frozenset({"autoregressive", "evolution"})
BASELINE_DOMAINS = frozenset({"vae_latent", "diffusion"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_eligible_master() -> tuple[pd.DataFrame, dict[str, str]]:
    master_path = ROOT / "outputs/master_scored_candidates.csv"
    manifest_path = ROOT / "shared-evaluator/reports/master_evaluation_manifest.json"
    source_paths = {
        "autoregressive": ROOT / "outputs/ar_candidates.csv",
        "vae_latent": ROOT / "outputs/vae_candidates.csv",
        "diffusion": ROOT / "outputs/diffusion_candidates.csv",
        "evolution": ROOT / "outputs/evolution_candidates.csv",
    }
    manifest = json.loads(manifest_path.read_text())
    master_hash = sha256_file(master_path)
    if master_hash != manifest.get("output_sha256"):
        raise ValueError("Role 06 master table differs from its evaluation manifest")
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    if source_hashes != manifest.get("input_sha256"):
        raise ValueError("Role 06 source tables differ from their evaluation manifest")
    df = pd.read_csv(master_path)
    required = {
        "sequence", "primary_domain", "contributing_domains", "is_valid",
        "synthesizable", "passes_novelty_rule_le80", "max_reference_similarity",
        "apex_mean_mic",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Role 06 master table lacks {sorted(missing)}")
    if df["contributing_domains"].isna().any():
        raise ValueError("Source attribution is missing")

    def allowed_domains(value: str) -> bool:
        domains = {domain.strip() for domain in value.split(";")}
        return bool(domains) and domains <= ENTRY_DOMAINS and not domains & BASELINE_DOMAINS

    source_mask = df["contributing_domains"].astype(str).map(allowed_domains)
    source_mask &= df["primary_domain"].isin(ENTRY_DOMAINS)
    valid_mask = df["is_valid"] & df["synthesizable"] & df["passes_novelty_rule_le80"]
    mic = pd.to_numeric(df["apex_mean_mic"], errors="coerce")
    valid_mask &= np.isfinite(mic) & (mic > 0)
    similarity = pd.to_numeric(df["max_reference_similarity"], errors="coerce")
    valid_mask &= np.isfinite(similarity) & similarity.between(0, 0.80, inclusive="both")
    eligible = df.loc[source_mask & valid_mask].copy().reset_index(drop=True)
    if len(eligible) < 100 or set(eligible["primary_domain"]) != ENTRY_DOMAINS:
        raise ValueError("Eligible master does not contain 100 candidates across both methods")
    return eligible, {
        "master_sha256": master_hash,
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_hashes,
        "eligible_rows": int(len(eligible)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Unique identifier for isolated outputs")
    args = parser.parse_args()
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, underscores, and hyphens")
    out_dir = ROOT / "portfolio-selection/outputs" / f"challenger_{args.run_id}"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite challenger outputs: {out_dir}")

    eligible, provenance = load_eligible_master()
    archive = MapElitesArchive().build_archive(eligible)
    eligible_annotated = archive["annotated_df"]
    optimizer = DppPortfolioOptimizer(
        alpha=0.60,
        rbf_sigma=2.0,
        det_weight=0.50,
        category_weight=0.30,
        max_domain_quota=60,
        min_domain_quota=1,
        max_pairwise_cosine=0.92,
        max_pairwise_levenshtein=0.80,
        required_domains=ENTRY_DOMAINS,
    )
    result = optimizer.optimize_portfolio(eligible_annotated, top_k=100, nested_k=50)
    top100 = result["top100_df"]
    top50 = result["top50_df"]
    if not set(top100["primary_domain"]) <= ENTRY_DOMAINS:
        raise AssertionError("Ranked output contains a non-entry method")
    if not all(top100["contributing_domains"].map(
        lambda value: set(str(value).split(";")) <= ENTRY_DOMAINS
    )):
        raise AssertionError("Ranked output contains baseline-source attribution")
    if top100["sequence"].nunique() != 100 or list(top50["sequence"]) != list(top100.head(50)["sequence"]):
        raise AssertionError("Challenger Top 50 / Top 100 invariant failed")
    import itertools
    import Levenshtein

    max_internal_ratio = max(
        Levenshtein.ratio(left, right)
        for left, right in itertools.combinations(top100["sequence"], 2)
    )
    if max_internal_ratio > 0.80:
        raise AssertionError(f"Top-100 internal Levenshtein ratio exceeds 0.80: {max_internal_ratio}")

    top_sequences = set(top100["sequence"])
    reserves = eligible_annotated[~eligible_annotated["sequence"].isin(top_sequences)].copy()
    reserves = reserves.sort_values("lcb_quality", ascending=False, kind="stable").head(200)
    reserves = reserves.reset_index(drop=True)
    reserves["portfolio_rank"] = np.arange(101, 101 + len(reserves))
    reserves["reserve_tier"] = np.where(
        reserves["portfolio_rank"] <= 200, "Tier_1_Niche_Matched", "Tier_2_Contingency"
    )

    out_dir.mkdir(parents=True)
    output_paths = {
        "top100.csv": top100,
        "top50.csv": top50,
        "reserves.csv": reserves,
    }
    for name, table in output_paths.items():
        table.to_csv(out_dir / name, index=False)
    top100.to_parquet(out_dir / "top100.parquet", index=False)
    top50.to_parquet(out_dir / "top50.parquet", index=False)
    reserves.to_parquet(out_dir / "reserves.parquet", index=False)
    report = {
        "run_id": args.run_id,
        "method": "fine_tuned_progen2_plus_project_evolutionary_search",
        "excluded_official_checkpoint_domains": sorted(BASELINE_DOMAINS),
        "provenance": provenance,
        "selection_source_sha256": {
            "dpp_optimizer.py": sha256_file(ROOT / "portfolio-selection/src/portfolio/dpp_optimizer.py"),
            "map_elites.py": sha256_file(ROOT / "portfolio-selection/src/portfolio/map_elites.py"),
            "run_submission_portfolio.py": sha256_file(Path(__file__)),
        },
        "archive_cells_occupied": archive["occupied_cells"],
        "archive_cells_total": archive["total_cells"],
        "domain_distribution": {name: int(count) for name, count in result["domain_distribution"].items()},
        "max_internal_top100_levenshtein_ratio": max_internal_ratio,
        "outputs_sha256": {
            name: sha256_file(out_dir / name)
            for name in [*output_paths, "top100.parquet", "top50.parquet", "reserves.parquet"]
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(out_dir), "eligible_rows": len(eligible), "domains": report["domain_distribution"], "sha256": report["outputs_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
