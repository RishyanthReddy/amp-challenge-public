"""Build, verify, and optionally promote the conservative local submission artifacts.

The ProGen2 auxiliary CSVs must already have been generated on the pinned Beam GPU and
verified against their run manifests. This workflow then selects the AR/evolution portfolio,
assembles and scores its 50,000-sequence library, renders FASTAs, and runs the independent
submission verifier. It never writes to a remote service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.staging")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def write_current_reports(challenger: Path, args: argparse.Namespace, entry_manifest: dict) -> None:
    assembly = entry_manifest["assembly_manifest"]
    top100 = pd.read_parquet(challenger / "top100.parquet")
    reserve_count = len(pd.read_parquet(challenger / "reserves.parquet"))
    library = pd.read_parquet(challenger / "full_50k_library.parquet")

    assembly_audit = f"""# Submission library assembly audit

Run: `{args.run_id}`

This submission entry uses the conservative autoregressive/evolutionary composition. Any
candidate with HydrAMP or AMP-Diffusion ancestry was excluded. The library contains
{assembly['eligible_master_provenance']['eligible_rows']:,} unique, scored AR/evolution candidates
followed by {assembly['auxiliary_sequences_selected']:,} unique ProGen2 auxiliary sequences.
The auxiliary sequences were selected from {assembly['unique_nonreference_auxiliary_pool']:,}
unique non-reference attempts after the shared AMP/safety model and synthesis filter.

| Check | Result |
| --- | ---: |
| Library sequences | {assembly['actual_library_size']:,} |
| Unique library sequences | {assembly['unique_sequences']:,} |
| Synthesis-filter passes | {assembly['synthesis_filter_pass_count']:,} |
| Exact reference overlaps | {assembly['exact_reference_overlap_count']} |
| Unique Top 100 | {assembly['top100_count']} |
| Top 100 subset of library | {assembly['top100_is_subset']} |
| Top-100 max similarity to references | {assembly['top100_max_reference_levenshtein_ratio']:.6f} |
| Top-100 maximum internal similarity | {assembly['top100_max_internal_levenshtein_ratio']:.6f} |
| Reserve rows | {reserve_count} |

The {len(library):,}-row library is screened with the project synthesis filter. Its AMP
probabilities and toxicity estimates are model predictions; auxiliary APEX MIC is not
available and no output is a wet-lab assay result. Source attribution and input hashes are
recorded in `library_assembly_manifest.json` and `submission_entry_manifest.json`.
"""
    (ROOT / "portfolio-selection/reports/input_and_library_assembly_audit.md").write_text(assembly_audit)

    from scipy.stats import hypergeom
    import numpy as np

    rows = []
    for label, table_path in (("Top 50", challenger / "top50.parquet"), ("Top 100", challenger / "top100.parquet")):
        table = pd.read_parquet(table_path)
        N = len(table)
        draw_k = min(25, N)
        model_candidate_count = int(((table["pred_amp_probability"] >= 0.70) & table["synthesizable"].astype(bool)).sum())
        exact_probabilities = {
            threshold: float(hypergeom.sf(threshold - 1, N, model_candidate_count, draw_k))
            for threshold in (1, 5, 10)
        }
        indicator = (
            (table["pred_amp_probability"].to_numpy(dtype=float) >= 0.70)
            & table["synthesizable"].astype(bool).to_numpy()
        )
        rng = np.random.default_rng(20260928)
        hits = np.empty(10_000, dtype=np.int16)
        for draw_index in range(len(hits)):
            picks = rng.choice(N, size=draw_k, replace=False)
            hits[draw_index] = int(indicator[picks].sum())
        mc = {
            threshold: float(np.mean(hits >= threshold))
            for threshold in (1, 5, 10)
        }
        mc_error = {
            threshold: abs(mc[threshold] - exact_probabilities[threshold])
            for threshold in (1, 5, 10)
        }
        mc_4se_limit = {
            threshold: 4 * float(np.sqrt(
                exact_probabilities[threshold] * (1 - exact_probabilities[threshold]) / len(hits)
            ))
            for threshold in (1, 5, 10)
        }
        if any(mc_error[k] > max(mc_4se_limit[k], 1 / len(hits)) for k in mc_error):
            raise AssertionError(f"Seeded Monte Carlo check is outside 4-SE tolerance for {label}")
        rows.append({
            "portfolio": label,
            "N": N,
            "model_predicted_candidates": model_candidate_count,
            "draw_k": draw_k,
            "expected_model_predicted": draw_k * model_candidate_count / N,
            **{f"exact_probability_at_least_{k}": value for k, value in exact_probabilities.items()},
            **{f"mc_probability_at_least_{k}": value for k, value in mc.items()},
            **{f"mc_abs_error_at_least_{k}": value for k, value in mc_error.items()},
            **{f"mc_four_se_limit_at_least_{k}": value for k, value in mc_4se_limit.items()},
            "mc_draws": len(hits),
            "mc_seed": 20260928,
        })
    lottery = "# Exact finite-population draw probabilities\n\n"
    lottery += "Hypergeometric values are exact for a uniform draw without replacement. The seeded 10,000-draw Monte Carlo estimates are compared with the exact values; each error must be within four binomial standard errors (or one simulation step). These are probabilities over model-threshold labels, not assay-hit probabilities. The threshold is predicted AMP probability ≥ 0.70 plus the synthesis-filter flag.\n\n"
    lottery += "| Portfolio | N | Threshold-positive | Draw k | Expected positives | Exact P(≥1) | MC P(≥1) | Error | Exact P(≥5) | MC P(≥5) | Exact P(≥10) | MC P(≥10) |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for row in rows:
        lottery += (
            f"| {row['portfolio']} | {row['N']} | {row['model_predicted_candidates']} | {row['draw_k']} | "
            f"{row['expected_model_predicted']:.3f} | {row['exact_probability_at_least_1']:.6f} | "
            f"{row['mc_probability_at_least_1']:.6f} | {row['mc_abs_error_at_least_1']:.6f} | "
            f"{row['exact_probability_at_least_5']:.6f} | {row['mc_probability_at_least_5']:.6f} | "
            f"{row['exact_probability_at_least_10']:.6f} | {row['mc_probability_at_least_10']:.6f} |\n"
        )
    lottery += "\n## Convergence check by event threshold\n\n| Portfolio | Event | Exact probability | Monte Carlo probability | Absolute error | Four standard errors | Pass |\n| --- | ---: | ---: | ---: | ---: | ---: | :---: |\n"
    for row in rows:
        for threshold in (1, 5, 10):
            exact = row[f"exact_probability_at_least_{threshold}"]
            error = row[f"mc_abs_error_at_least_{threshold}"]
            limit = row[f"mc_four_se_limit_at_least_{threshold}"]
            tolerance = max(limit, 1 / row["mc_draws"])
            lottery += (
                f"| {row['portfolio']} | ≥{threshold} | {exact:.6f} | "
                f"{row[f'mc_probability_at_least_{threshold}']:.6f} | {error:.6f} | "
                f"{limit:.6f} | {'PASS' if error <= tolerance else 'FAIL'} |\n"
            )
    (ROOT / "portfolio-selection/reports/portfolio_lottery_report.md").write_text(lottery)
    lottery_summary = pd.DataFrame(rows)
    lottery_summary.to_csv(ROOT / "portfolio-selection/outputs/lottery_simulation_summary.csv", index=False)
    lottery_summary.to_parquet(ROOT / "portfolio-selection/outputs/lottery_simulation_summary.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--aux-csv", type=Path, action="append", required=True)
    parser.add_argument(
        "--promote-canonical", action="store_true",
        help="After challenger verification, replace the local canonical portfolio and FASTAs.",
    )
    args = parser.parse_args()
    challenger = ROOT / "portfolio-selection/outputs" / f"challenger_{args.run_id}"
    if not challenger.is_dir():
        run(["uv", "run", "--project", "portfolio-selection", "python",
             str(ROOT / "portfolio-selection/scripts/run_submission_portfolio.py"),
             "--run-id", args.run_id])
    else:
        raise FileExistsError(f"Refusing to reuse existing challenger output: {challenger}")

    aux_paths = []
    for path in args.aux_csv:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        aux_paths.extend(["--aux-csv", str(resolved)])
    run([
        "uv", "run", "--project", "shared-evaluator", "python",
        str(ROOT / "portfolio-selection/scripts/assemble_submission_library.py"),
        "--run-id", args.run_id,
        "--challenger-dir", str(challenger),
        *aux_paths,
    ])

    challenger_fasta = challenger / "fasta"
    run([
        "uv", "run", "export_submission",
        "--top-parquet", str(challenger / "top100.parquet"),
        "--library-parquet", str(challenger / "full_50k_library.parquet"),
        "--output-dir", str(challenger_fasta),
    ])
    run([
        "uv", "run", "python", str(ROOT / "scripts/verify_submission.py"), str(ROOT),
        "--library-fasta", str(challenger_fasta / "library.fasta"),
        "--top-fasta", str(challenger_fasta / "top.fasta"),
        "--no-replay",
    ])

    if args.promote_canonical:
        # Keep a local copy of the superseded research snapshot before promotion.
        archive = ROOT / "submission_preparation/research_four_domain_20260927"
        if archive.exists():
            raise FileExistsError(f"Research snapshot archive already exists: {archive}")
        archive.mkdir(parents=True)
        for relative in (
            "portfolio-selection/outputs/full_50k_library.parquet",
            "portfolio-selection/outputs/portfolio_top100.parquet",
            "portfolio-selection/outputs/portfolio_top50.parquet",
            "portfolio-selection/outputs/portfolio_reserves.parquet",
            "portfolio-selection/reports/portfolio_run_manifest.json",
            "portfolio-selection/reports/library_assembly_manifest.json",
            "portfolio-selection/reports/submission_entry_manifest.json",
            "portfolio-selection/reports/submission_artifact_verification.json",
            "portfolio-selection/reports/input_and_library_assembly_audit.md",
            "portfolio-selection/reports/portfolio_lottery_report.md",
            "portfolio-selection/outputs/lottery_simulation_summary.csv",
            "portfolio-selection/outputs/lottery_simulation_summary.parquet",
            "generate_broad_spectrum/library.fasta",
            "generate_broad_spectrum/top.fasta",
            "generate/library.fasta",
            "generate/top.fasta",
            "outputs/portfolio_top100.csv",
            "outputs/portfolio_top50.csv",
            "outputs/portfolio_reserves.csv",
        ):
            source = ROOT / relative
            if source.is_file():
                target = archive / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        archive_manifest = {
            "description": "Four-domain research snapshot preserved before conservative AR/evolution promotion.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files_sha256": {
                path.relative_to(archive).as_posix(): sha256(path)
                for path in archive.rglob("*") if path.is_file()
            },
        }
        (archive / "snapshot_manifest.json").write_text(
            json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n"
        )

        for name in ("full_50k_library.parquet",):
            copy_atomic(challenger / name, ROOT / "portfolio-selection/outputs" / name)
        for name, source_name in (
            ("portfolio_top100.parquet", "top100.parquet"),
            ("portfolio_top50.parquet", "top50.parquet"),
            ("portfolio_reserves.parquet", "reserves.parquet"),
        ):
            copy_atomic(challenger / source_name, ROOT / "portfolio-selection/outputs" / name)
        for name, source_name in (
            ("portfolio_top100.csv", "top100.csv"),
            ("portfolio_top50.csv", "top50.csv"),
            ("portfolio_reserves.csv", "reserves.csv"),
        ):
            copy_atomic(challenger / source_name, ROOT / "portfolio-selection/outputs" / name)
            copy_atomic(challenger / source_name, ROOT / "outputs" / name)

        run(["uv", "run", "export_submission"])
        run(["uv", "run", "python", "scripts/verify_submission.py", ".", "--no-replay"])
        for name in ("library.fasta", "top.fasta"):
            if sha256(ROOT / "generate_broad_spectrum" / name) != sha256(challenger_fasta / name):
                raise RuntimeError(f"Promoted canonical FASTA differs from verified challenger: {name}")

        # Synchronize the global Role 07 and submission reports with promoted outputs.
        challenger_manifest = json.loads((challenger / "manifest.json").read_text())
        assembly_manifest = json.loads((challenger / "assembly_manifest.json").read_text())
        canonical_report = dict(challenger_manifest)
        canonical_report.update({
            "role": "07_portfolio_selection",
            "canonical_submission_run_id": args.run_id,
            "master_candidate_sha256": challenger_manifest["provenance"]["master_sha256"],
            "role06_manifest_sha256": challenger_manifest["provenance"]["evaluation_manifest_sha256"],
            "role06_input_sha256": challenger_manifest["provenance"]["source_sha256"],
            "candidate_rows_with_apex_scores": int(challenger_manifest["provenance"]["eligible_rows"]),
            "top100_rows": 100,
            "top50_rows": 50,
            "reserves_rows": len(pd.read_parquet(challenger / "reserves.parquet")),
            "top100_nested_top50": True,
            "canonical_top100_sha256": sha256(ROOT / "portfolio-selection/outputs/portfolio_top100.parquet"),
            "output_sha256": {
                "portfolio_top100.parquet": sha256(ROOT / "portfolio-selection/outputs/portfolio_top100.parquet"),
                "portfolio_top50.parquet": sha256(ROOT / "portfolio-selection/outputs/portfolio_top50.parquet"),
                "portfolio_reserves.parquet": sha256(ROOT / "portfolio-selection/outputs/portfolio_reserves.parquet"),
            },
            "domain_distribution": challenger_manifest["domain_distribution"],
        })
        (ROOT / "portfolio-selection/reports/portfolio_run_manifest.json").write_text(
            json.dumps(canonical_report, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(challenger / "assembly_manifest.json", ROOT / "portfolio-selection/reports/library_assembly_manifest.json")
        entry_manifest = {
            "run_id": args.run_id,
            "method": "fine-tuned ProGen2 plus project evolutionary search; official HydrAMP and AMP-Diffusion checkpoint outputs excluded",
            "working_rule_interpretation": "One integrated generative entry using two distinct participant-developed methods; organizer interpretation remains unconfirmed.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "challenger_directory": str(challenger.relative_to(ROOT)),
            "portfolio_manifest_sha256": sha256(challenger / "manifest.json"),
            "assembly_manifest_sha256": sha256(challenger / "assembly_manifest.json"),
            "portfolio_manifest": canonical_report,
            "assembly_manifest": assembly_manifest,
            "canonical_files_sha256": {
                name: sha256(ROOT / name) for name in (
                    "generate_broad_spectrum/library.fasta",
                    "generate_broad_spectrum/top.fasta",
                    "portfolio-selection/outputs/full_50k_library.parquet",
                    "portfolio-selection/outputs/portfolio_top100.parquet",
                    "portfolio-selection/outputs/portfolio_top50.parquet",
                    "portfolio-selection/outputs/portfolio_reserves.parquet",
                )
            },
            "verification": {
                "library_count": 50_000,
                "top100_count": 100,
                "unique_library": True,
                "unique_top100": True,
                "top100_subset": True,
                "exact_reference_overlap": 0,
                "max_reference_similarity": float(assembly_manifest.get("top100_max_reference_levenshtein_ratio", 0.0)),
                "max_internal_similarity": float(assembly_manifest["top100_max_internal_levenshtein_ratio"]),
                "all_library_sequences_synthesizable": assembly_manifest["synthesis_filter_pass_count"] == 50_000,
            },
            "promotion_archive": str(archive.relative_to(ROOT)),
            "release_limitations": [
                "Baseline-policy classification is a conservative interpretation, not an organizer ruling.",
                "Public redistribution rights for source datasets and assembled training data remain unresolved.",
                "The submission command replays frozen outputs; it does not retrain generators or evaluators.",
            ],
        }
        entry_path = ROOT / "portfolio-selection/reports/submission_entry_manifest.json"
        entry_path.write_text(json.dumps(entry_manifest, indent=2, sort_keys=True) + "\n")
        verification_path = ROOT / "portfolio-selection/reports/submission_artifact_verification.json"
        verification_path.write_text(json.dumps({
            "run_id": args.run_id,
            "status": "PASS",
            **entry_manifest["verification"],
            "canonical_files_sha256": entry_manifest["canonical_files_sha256"],
            "independent_verifier": "scripts/verify_submission.py",
        }, indent=2, sort_keys=True) + "\n")
        write_current_reports(challenger, args, entry_manifest)

    print(json.dumps({
        "challenger_directory": str(challenger),
        "library_sha256": sha256(challenger_fasta / "library.fasta"),
        "top_sha256": sha256(challenger_fasta / "top.fasta"),
        "promoted_canonical": bool(args.promote_canonical),
    }, indent=2))


if __name__ == "__main__":
    main()
