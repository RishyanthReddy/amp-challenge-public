"""Use the repository's canonical verifier and save fresh Role 07 artifact hashes."""

import hashlib
import itertools
import json
import csv
import math
import sys
from pathlib import Path

import pandas as pd
import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_submission import _read_fasta, verify_submission
sys.path.insert(0, str(ROOT / "data-engineering/src"))
from amp_data.synthesis_filter import is_synthesizable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_file(relative_path: str) -> Path:
    path = Path(str(relative_path))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe repository-relative path in manifest: {relative_path!r}")
    return ROOT / path


def validate_auxiliary_run(run_record: dict) -> dict:
    """Recheck the hash/count chain from raw ProGen2 output through PPL scoring."""
    scored_csv = repo_file(run_record["csv"])
    scored_manifest_path = repo_file(run_record["manifest"])
    if sha256_file(scored_csv) != run_record["csv_sha256"]:
        raise ValueError(f"Scored auxiliary CSV hash mismatch: {scored_csv}")
    if sha256_file(scored_manifest_path) != run_record["manifest_sha256"]:
        raise ValueError(f"Scored auxiliary manifest hash mismatch: {scored_manifest_path}")
    scored_manifest = json.loads(scored_manifest_path.read_text())
    if scored_manifest.get("csv_sha256") != run_record["csv_sha256"]:
        raise ValueError(f"Scored auxiliary manifest points at a different CSV: {scored_manifest_path}")
    if scored_manifest.get("run_id") != run_record["run_id"]:
        raise ValueError(f"Scored auxiliary run ID mismatch: {scored_manifest_path}")

    raw_csv = repo_file(scored_manifest["source_csv"])
    raw_manifest_path = raw_csv.with_suffix(".manifest.json")
    if sha256_file(raw_csv) != scored_manifest.get("raw_source_csv_sha256"):
        raise ValueError(f"Raw auxiliary CSV hash mismatch: {raw_csv}")
    if sha256_file(raw_manifest_path) != scored_manifest.get("raw_source_manifest_sha256"):
        raise ValueError(f"Raw auxiliary manifest hash mismatch: {raw_manifest_path}")
    raw_manifest = json.loads(raw_manifest_path.read_text())
    if raw_manifest.get("csv_sha256") != scored_manifest.get("raw_source_csv_sha256"):
        raise ValueError(f"Raw auxiliary manifest disagrees with its CSV: {raw_manifest_path}")
    if (
        raw_manifest.get("run_id") != scored_manifest.get("run_id")
        or int(raw_manifest.get("n_attempts", -1)) != int(scored_manifest.get("n_generation_attempts", -2))
        or int(raw_manifest.get("n_valid", -1)) != int(scored_manifest.get("n_generation_valid", -2))
        or int(raw_manifest.get("n_unique_valid", -1)) != int(scored_manifest.get("n_unique_valid", -2))
        or scored_manifest.get("runner_source_sha256") != run_record.get("runner_source_sha256")
        or scored_manifest.get("runner_source_file") != run_record.get("runner_source_file")
        or float(scored_manifest.get("max_perplexity", -1)) != float(run_record.get("max_perplexity", -2))
        or float(scored_manifest.get("max_observed_perplexity", -1)) != float(run_record.get("max_observed_perplexity", -2))
        or int(scored_manifest.get("n_perplexity_pass", -1)) != int(run_record.get("n_perplexity_pass", -2))
        or scored_manifest.get("perplexity_runner_source_sha256") != run_record.get("perplexity_runner_source_sha256")
        or scored_manifest.get("perplexity_runner_source_file") != run_record.get("perplexity_runner_source_file")
    ):
        raise ValueError(f"Scored and raw auxiliary run provenance disagree: {scored_manifest_path}")

    for file_key, hash_key in (
        ("runner_source_file", "runner_source_sha256"),
        ("perplexity_runner_source_file", "perplexity_runner_source_sha256"),
    ):
        source_path = repo_file(scored_manifest[file_key])
        if sha256_file(source_path) != scored_manifest[hash_key]:
            raise ValueError(f"Auxiliary source snapshot hash mismatch: {source_path}")

    threshold = float(scored_manifest["max_perplexity"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"Invalid auxiliary perplexity threshold: {scored_manifest_path}")
    attempts = valid = unique_valid = ppl_pass = 0
    sequences = set()
    with scored_csv.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        required = {"attempt_index", "sequence", "is_valid", "perplexity", "passes_perplexity_filter"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"PPL-scored CSV has an incomplete schema: {scored_csv}")
        for row in reader:
            attempts += 1
            if int(row["attempt_index"]) != attempts:
                raise ValueError(f"Auxiliary attempt indices are missing or unordered: {scored_csv}")
            is_valid = row["is_valid"] == "True"
            valid += int(is_valid)
            if is_valid:
                sequences.add(row["sequence"].strip().upper())
            perplexity = float(row["perplexity"])
            passed = row["passes_perplexity_filter"] == "True"
            if passed != bool(is_valid and math.isfinite(perplexity) and perplexity <= threshold):
                raise ValueError(f"Auxiliary PPL decision is inconsistent with its score: {scored_csv}")
            ppl_pass += int(passed)
    unique_valid = len(sequences)

    expected = {
        "attempts": attempts,
        "valid_attempts": valid,
        "unique_valid_attempts": unique_valid,
        "n_perplexity_pass": ppl_pass,
    }
    if (
        attempts != int(scored_manifest["n_attempts"])
        or valid != int(scored_manifest["n_valid"])
        or unique_valid != int(scored_manifest["n_unique_valid"])
        or ppl_pass != int(scored_manifest["n_perplexity_pass"])
        or expected["attempts"] != int(run_record["attempts"])
        or expected["valid_attempts"] != int(run_record["valid_attempts"])
        or expected["unique_valid_attempts"] != int(run_record["unique_valid_attempts"])
        or expected["n_perplexity_pass"] != int(run_record["n_perplexity_pass"])
    ):
        raise ValueError(f"Auxiliary manifest counts disagree with scored CSV: {scored_csv}")
    return {
        "run_id": run_record["run_id"],
        "csv_sha256": run_record["csv_sha256"],
        "manifest_sha256": run_record["manifest_sha256"],
        **expected,
    }


def main() -> None:
    verify_submission(ROOT, replay=False)
    library_path = ROOT / "generate_broad_spectrum/library.fasta"
    top_path = ROOT / "generate_broad_spectrum/top.fasta"
    master_path = ROOT / "outputs/master_scored_candidates.csv"
    role06_manifest_path = ROOT / "shared-evaluator/reports/master_evaluation_manifest.json"
    portfolio_manifest_path = ROOT / "portfolio-selection/reports/portfolio_run_manifest.json"
    assembly_manifest_path = ROOT / "portfolio-selection/reports/library_assembly_manifest.json"
    entry_manifest_path = ROOT / "portfolio-selection/reports/submission_entry_manifest.json"
    top_parquet_path = ROOT / "portfolio-selection/outputs/portfolio_top100.parquet"
    library_parquet_path = ROOT / "portfolio-selection/outputs/full_50k_library.parquet"
    for path in (
        master_path, role06_manifest_path, portfolio_manifest_path,
        assembly_manifest_path, top_parquet_path, library_parquet_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required current Role 07 provenance is missing: {path}")

    role06_manifest = json.loads(role06_manifest_path.read_text())
    portfolio_manifest = json.loads(portfolio_manifest_path.read_text())
    assembly_manifest = json.loads(assembly_manifest_path.read_text())
    entry_manifest = json.loads(entry_manifest_path.read_text()) if entry_manifest_path.is_file() else None
    master_sha = sha256_file(master_path)
    top_parquet_sha = sha256_file(top_parquet_path)
    library_parquet_sha = sha256_file(library_parquet_path)
    if role06_manifest.get("output_sha256") != master_sha:
        raise ValueError("Current master table hash disagrees with Role 06 manifest")
    if portfolio_manifest.get("master_candidate_sha256") != master_sha:
        raise ValueError("Current portfolio was not built from the current master table")
    if portfolio_manifest.get("role06_manifest_sha256") != sha256_file(role06_manifest_path):
        raise ValueError("Portfolio manifest does not match the current Role 06 manifest")
    if portfolio_manifest.get("output_sha256", {}).get("portfolio_top100.parquet") != top_parquet_sha:
        raise ValueError("Top-100 Parquet hash disagrees with the Role 07 portfolio manifest")
    if "eligible_master_provenance" in assembly_manifest:
        master_provenance = assembly_manifest["eligible_master_provenance"]
        if master_provenance.get("master_sha256") != master_sha:
            raise ValueError("Library assembly manifest does not match the current Role 06 master")
        if master_provenance.get("evaluation_manifest_sha256") != sha256_file(role06_manifest_path):
            raise ValueError("Library assembly manifest does not match the current Role 06 manifest")
        if assembly_manifest.get("challenger_top100_sha256") != top_parquet_sha:
            raise ValueError("Library assembly manifest does not match the selected Top 100")
        if entry_manifest is None:
            raise FileNotFoundError("Current conservative assembly requires a submission-entry manifest")
        challenger_dir = repo_file(entry_manifest["challenger_directory"])
        challenger_manifest_path = challenger_dir / "manifest.json"
        challenger_assembly_path = challenger_dir / "assembly_manifest.json"
        if sha256_file(challenger_manifest_path) != entry_manifest.get("portfolio_manifest_sha256"):
            raise ValueError("Submission entry points to a different challenger portfolio manifest")
        if sha256_file(challenger_assembly_path) != entry_manifest.get("assembly_manifest_sha256"):
            raise ValueError("Submission entry points to a different challenger assembly manifest")
        if sha256_file(assembly_manifest_path) != entry_manifest.get("assembly_manifest_sha256"):
            raise ValueError("Canonical library assembly manifest hash differs from the submission entry")
        if sha256_file(challenger_manifest_path) != assembly_manifest.get("challenger_manifest_sha256"):
            raise ValueError("Library assembly manifest does not match its challenger portfolio manifest")
        if json.loads(challenger_assembly_path.read_text()) != assembly_manifest:
            raise ValueError("Canonical and challenger library assembly manifests differ")
        if entry_manifest.get("assembly_manifest") != assembly_manifest:
            raise ValueError("Submission entry embeds a different library assembly manifest")
        if entry_manifest.get("portfolio_manifest_sha256") != sha256_file(challenger_manifest_path):
            raise ValueError("Submission entry portfolio hash is inconsistent")
        if assembly_manifest.get("reference_fasta_sha256") != sha256_file(ROOT / "data-engineering/data/challenge/antibacterial.fasta"):
            raise ValueError("Library assembly manifest uses a different reference FASTA")
        if assembly_manifest.get("exact_reference_overlap_count") != 0:
            raise ValueError("Library assembly manifest records an exact reference overlap")
        validated_runs = [
            validate_auxiliary_run(run_record)
            for run_record in assembly_manifest.get("auxiliary_provenance", {}).get("runs", [])
        ]
        if len(validated_runs) != len(assembly_manifest.get("auxiliary_provenance", {}).get("runs", [])):
            raise ValueError("Auxiliary run provenance is missing")
    else:
        validated_runs = []
        if assembly_manifest.get("master_candidate_sha256") != master_sha:
            raise ValueError("Library assembly manifest does not match the current master table")
        if assembly_manifest.get("role07_portfolio_manifest_sha256") != sha256_file(portfolio_manifest_path):
            raise ValueError("Library assembly manifest does not match the current portfolio manifest")
    if assembly_manifest.get("library_parquet_sha256") != library_parquet_sha:
        if assembly_manifest.get("library_sha256") != library_parquet_sha:
            raise ValueError("Library Parquet hash disagrees with its assembly manifest")

    df_library = pd.read_parquet(library_parquet_path).sort_values("library_index", kind="stable")
    df_top = pd.read_parquet(top_parquet_path).sort_values("portfolio_rank", kind="stable")
    if len(df_library) != 50_000 or df_library["sequence"].nunique() != 50_000:
        raise ValueError("Current assembled library Parquet does not contain 50,000 unique sequences")
    if len(df_top) != 100 or df_top["sequence"].nunique() != 100:
        raise ValueError("Current Top-100 Parquet does not contain 100 unique sequences")
    library_headers, fasta_library = _read_fasta(library_path)
    top_headers, fasta_top = _read_fasta(top_path)
    expected_library = df_library["sequence"].astype(str).str.upper().tolist()
    expected_top = df_top["sequence"].astype(str).str.upper().tolist()
    if fasta_library != expected_library or fasta_top != expected_top:
        raise ValueError("Submission FASTAs do not serialize the current Role 07 Parquet rows in order")
    if not set(expected_top).issubset(set(expected_library)):
        raise ValueError("Current Top 100 is not a subset of the current library")
    if not portfolio_manifest.get("top100_nested_top50"):
        raise ValueError("The current Top 50 is not nested in the Top 100")

    if entry_manifest is not None:
        for relative_path, expected_sha in entry_manifest.get("canonical_files_sha256", {}).items():
            if sha256_file(repo_file(relative_path)) != expected_sha:
                raise ValueError(f"Canonical output hash differs from the submission entry: {relative_path}")
    if "eligible_master_provenance" in assembly_manifest:
        entry_domains = {"autoregressive", "evolution"}
        if set(df_top["primary_domain"]) != entry_domains:
            raise ValueError("Current Top 100 does not contain exactly the selected AR/evolution domains")
        if not all(set(str(value).split(";")) <= entry_domains for value in df_top["contributing_domains"]):
            raise ValueError("Current Top 100 contains excluded baseline ancestry")
        if not set(df_library["primary_domain"]) <= entry_domains:
            raise ValueError("Current library contains a sequence from an excluded baseline domain")
        if (
            assembly_manifest["eligible_master_provenance"]["eligible_rows"]
            + assembly_manifest["auxiliary_sequences_selected"] != 50_000
        ):
            raise ValueError("Library source counts do not sum to 50,000")
        if assembly_manifest["synthesis_filter_pass_count"] != 50_000:
            raise ValueError("Assembly manifest does not report synthesis passage for the whole library")
        non_synthesizable = [
            sequence for sequence in df_library["sequence"]
            if not is_synthesizable(str(sequence))[0]
        ]
        if non_synthesizable:
            raise ValueError(f"Current library contains {len(non_synthesizable)} synthesis-filter failures")
        for key, relative in (
            ("dpp_optimizer.py", "portfolio-selection/src/portfolio/dpp_optimizer.py"),
            ("map_elites.py", "portfolio-selection/src/portfolio/map_elites.py"),
            ("run_submission_portfolio.py", "portfolio-selection/scripts/run_submission_portfolio.py"),
        ):
            if sha256_file(ROOT / relative) != portfolio_manifest["selection_source_sha256"].get(key):
                raise ValueError(f"Current portfolio selector source hash changed: {relative}")

    max_internal_similarity = max(
        (Levenshtein.ratio(left, right) for left, right in itertools.combinations(fasta_top, 2)),
        default=0.0,
    )
    manifest = {
        "role": "07_portfolio_and_submission_artifacts",
        "run_id": assembly_manifest.get("run_id", portfolio_manifest.get("run_id")),
        "method": assembly_manifest.get("method", portfolio_manifest.get("method")),
        "library_sha256": sha256_file(library_path),
        "top_sha256": sha256_file(top_path),
        "master_candidate_sha256": master_sha,
        "role06_manifest_sha256": sha256_file(role06_manifest_path),
        "portfolio_manifest_sha256": sha256_file(portfolio_manifest_path),
        "library_assembly_manifest_sha256": sha256_file(assembly_manifest_path),
        "library_parquet_sha256": library_parquet_sha,
        "top100_parquet_sha256": top_parquet_sha,
        "library_records": 50_000,
        "top_records": 100,
        "top_is_strict_subset": True,
        "top50_nested_in_top100": True,
        "top100_domain_distribution": portfolio_manifest["domain_distribution"],
        "max_internal_top100_levenshtein_ratio": max_internal_similarity,
        "max_reference_top100_levenshtein_ratio": assembly_manifest.get("top100_max_reference_levenshtein_ratio"),
        "exact_reference_overlap_count": assembly_manifest.get("exact_reference_overlap_count", 0),
        "all_library_sequences_synthesizable": assembly_manifest.get("synthesis_filter_pass_count") == 50_000,
        "auxiliary_runs_validated": validated_runs,
        "top_headers_clean": all("nan" not in header.lower() for header in top_headers),
        "library_headers_clean": all("nan" not in header.lower() for header in library_headers),
        "verifier": "scripts/verify_submission.py",
        "reference_sha256": sha256_file(ROOT / "data-engineering/data/challenge/antibacterial.fasta"),
    }
    out_path = ROOT / "portfolio-selection/reports/submission_artifact_verification.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Current Role 07 artifact hashes saved to {out_path}")


if __name__ == "__main__":
    main()
