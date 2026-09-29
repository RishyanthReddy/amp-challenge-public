"""Assemble the conservative AR/evolution submission library with scored ProGen2 fillers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "portfolio-selection/scripts"))
sys.path.insert(0, str(ROOT / "data-engineering/src"))
sys.path.insert(0, str(ROOT / "shared-evaluator/src"))

from amp_data.core import fasta_rows, norm
from amp_data.synthesis_filter import is_synthesizable
from run_submission_portfolio import load_eligible_master
from evaluator.activity_safety_models import extract_features

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
LIBRARY_SIZE = 50_000
REFERENCE_COUNT = 39_448


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_sequences(sequences: list[str], model_dir: Path) -> pd.DataFrame:
    """Batch-score auxiliary peptides with the checked-in Role 06 models."""
    amp_model = __import__("joblib").load(model_dir / "amp_classifier_ensemble.joblib")
    hemo_model = __import__("joblib").load(model_dir / "empirical_hemolysis_ensemble.joblib")
    features = np.asarray([extract_features(sequence) for sequence in sequences], dtype=np.float32)
    amp_tree = np.column_stack([
        tree.predict_proba(features)[:, 1] for tree in amp_model.estimators_
    ])
    hemo_tree = np.column_stack([
        tree.predict_proba(features)[:, 1] for tree in hemo_model.estimators_
    ])
    amp_mean = amp_tree.mean(axis=1)
    hemo_mean = hemo_tree.mean(axis=1)
    hemo_std = hemo_tree.std(axis=1)
    moment = features[:, 23]
    gravy = features[:, 24]
    membrane_proxy = np.clip(0.5 * (gravy - 0.2) * 1.5 + 0.5 * (moment - 0.4) * 2.0, 0.0, 1.0)
    tox = 0.5 * hemo_mean + 0.5 * membrane_proxy
    return pd.DataFrame({
        "pred_amp_probability": amp_mean,
        "uncertainty_amp_std": amp_tree.std(axis=1),
        "pred_empirical_hemolysis_prob": hemo_mean,
        "uncertainty_hemo_std": hemo_std,
        "biophysical_membrane_disruption_proxy": membrane_proxy,
        "pred_toxicity_risk": tox,
        "library_selection_score": amp_mean - 0.5 * tox - amp_tree.std(axis=1),
    })


def load_auxiliary(paths: list[Path], reference: set[str], already_seen: set[str]) -> tuple[pd.DataFrame, dict]:
    records = []
    run_manifests = []
    rejected = {"invalid": 0, "perplexity": 0, "reference_overlap": 0, "duplicate": 0}
    for path in paths:
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing auxiliary generation manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        csv_hash = sha256_file(path)
        if manifest.get("csv_sha256") != csv_hash:
            raise ValueError(f"Auxiliary CSV hash does not match its manifest: {path}")
        if manifest.get("checkpoint_sha256") != "124b8ea7df5c96cda927bada51c9d26d89f91636d0975fe70e7bc0ef182f9f83":
            raise ValueError(f"Unexpected ProGen2 checkpoint in auxiliary run: {path}")
        if manifest.get("model_code_revision") != "43237a0b733c6629226a079266d2985c9fdce9b7":
            raise ValueError(f"Unexpected ProGen2 code revision in auxiliary run: {path}")
        if manifest.get("tokenizer_sha256") != "cc489cd8bfeab3c70c6a2954d2963b9fb8e7ee4b13aaa0e84e97d7f17f35d43c":
            raise ValueError(f"Unexpected ProGen2 tokenizer in auxiliary run: {path}")
        runner_source = ROOT / str(manifest.get("runner_source_file", ""))
        if not runner_source.is_file() or sha256_file(runner_source) != manifest.get("runner_source_sha256"):
            raise ValueError(f"Auxiliary runner source does not match its recorded hash: {path}")
        if manifest.get("generation_runner_source_sha256") != manifest.get("runner_source_sha256"):
            raise ValueError(f"Auxiliary generation-source hashes disagree: {path}")
        perplexity_runner = ROOT / str(manifest.get("perplexity_runner_source_file", ""))
        if (
            not perplexity_runner.is_file()
            or sha256_file(perplexity_runner) != manifest.get("perplexity_runner_source_sha256")
        ):
            raise ValueError(f"Auxiliary perplexity runner does not match its recorded hash: {path}")
        source_csv_value = Path(str(manifest.get("source_csv", "")))
        if source_csv_value.is_absolute() or ".." in source_csv_value.parts:
            raise ValueError(f"Unsafe raw auxiliary CSV path in manifest: {path}")
        raw_csv = ROOT / source_csv_value
        raw_manifest = raw_csv.with_suffix(".manifest.json")
        if (
            not raw_csv.is_file()
            or sha256_file(raw_csv) != manifest.get("raw_source_csv_sha256")
            or not raw_manifest.is_file()
            or sha256_file(raw_manifest) != manifest.get("raw_source_manifest_sha256")
        ):
            raise ValueError(f"Perplexity-scored auxiliary data cannot be traced to its raw run: {path}")
        raw_run_manifest = json.loads(raw_manifest.read_text())
        if raw_run_manifest.get("csv_sha256") != manifest.get("raw_source_csv_sha256"):
            raise ValueError(f"Raw auxiliary CSV hash disagrees with its generation manifest: {path}")
        if (
            raw_run_manifest.get("run_id") != manifest.get("run_id")
            or int(raw_run_manifest.get("n_attempts", -1)) != int(manifest.get("n_generation_attempts", -2))
            or int(raw_run_manifest.get("n_valid", -1)) != int(manifest.get("n_generation_valid", -2))
            or int(raw_run_manifest.get("n_unique_valid", -1)) != int(manifest.get("n_unique_valid", -2))
        ):
            raise ValueError(f"Perplexity manifest generation counts disagree with the raw run: {path}")
        ppl_threshold = float(manifest.get("max_perplexity", float("nan")))
        max_observed_ppl = float(manifest.get("max_observed_perplexity", float("nan")))
        ppl_elapsed = float(manifest.get("perplexity_elapsed_sec", float("nan")))
        ppl_rate = float(manifest.get("perplexity_attempts_per_sec", float("nan")))
        if (
            not np.isfinite([ppl_threshold, max_observed_ppl, ppl_elapsed, ppl_rate]).all()
            or ppl_threshold <= 0
            or max_observed_ppl < 0
            or ppl_elapsed <= 0
            or ppl_rate <= 0
        ):
            raise ValueError(f"Invalid perplexity metrics in auxiliary manifest: {path}")
        run_id = str(manifest.get("run_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError(f"Invalid run ID in auxiliary manifest: {run_id!r}")
        row_count = 0
        valid_count = 0
        perplexity_pass_count = 0
        run_unique_valid = set()
        with path.open(newline="", encoding="ascii") as stream:
            for row in csv.DictReader(stream):
                row_count += 1
                if int(row["attempt_index"]) != row_count:
                    raise ValueError(f"Attempt indices are missing or unordered in {path}")
                sequence = norm(row.get("sequence", ""))
                is_valid = row.get("is_valid") == "True"
                if is_valid:
                    valid_count += 1
                    run_unique_valid.add(sequence)
                if not is_valid or not (8 <= len(sequence) <= 50) or not set(sequence) <= STANDARD_AA:
                    rejected["invalid"] += 1
                    continue
                try:
                    perplexity = float(row["perplexity"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"PPL-scored auxiliary CSV is missing a valid perplexity: {path}") from exc
                threshold = ppl_threshold
                if (
                    not np.isfinite(perplexity)
                    or threshold <= 0
                    or row.get("passes_perplexity_filter") != "True"
                    or perplexity > threshold
                ):
                    rejected["perplexity"] += 1
                    continue
                perplexity_pass_count += 1
                if sequence in reference:
                    rejected["reference_overlap"] += 1
                    continue
                if sequence in already_seen:
                    rejected["duplicate"] += 1
                    continue
                already_seen.add(sequence)
                records.append({
                    "sequence_id": f"progen_{run_id}_{int(row['attempt_index']):06d}",
                    "sequence": sequence,
                    "length": len(sequence),
                    "source_tier": "progen2_auxiliary",
                    "primary_domain": "autoregressive",
                    "contributing_domains": "autoregressive",
                    "attempt_index": int(row["attempt_index"]),
                    "generation_run_id": run_id,
                    "perplexity": perplexity,
                })
        if row_count != int(manifest["n_attempts"]) or valid_count != int(manifest["n_valid"]):
            raise ValueError(f"Auxiliary CSV row counts disagree with its manifest: {path}")
        if len(run_unique_valid) != int(manifest["n_unique_valid"]):
            raise ValueError(f"Auxiliary sequence uniqueness count disagrees with its manifest: {path}")
        if perplexity_pass_count != int(manifest.get("n_perplexity_pass", -1)):
            raise ValueError(f"Auxiliary perplexity pass count disagrees with its manifest: {path}")
        run_manifests.append({
            "csv": str(path.relative_to(ROOT)),
            "csv_sha256": csv_hash,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "run_id": run_id,
            "attempts": int(manifest["n_attempts"]),
            "valid_attempts": int(manifest["n_valid"]),
            "unique_valid_attempts": int(manifest["n_unique_valid"]),
            "run_status": manifest.get("run_status", "completed"),
            "requested_attempts": int(manifest.get("requested_attempts", manifest["n_attempts"])),
            "generation_gpu": manifest.get("generation_gpu"),
            "generation_attempts_per_sec": (
                float(manifest["generation_attempts_per_sec"])
                if manifest.get("generation_attempts_per_sec") is not None
                else None
            ),
            "runner_source_sha256": manifest.get("runner_source_sha256"),
            "runner_source_file": str(runner_source.relative_to(ROOT)),
            "max_perplexity": ppl_threshold,
            "max_observed_perplexity": max_observed_ppl,
            "n_perplexity_pass": perplexity_pass_count,
            "perplexity_gpu": manifest["gpu"],
            "perplexity_elapsed_sec": ppl_elapsed,
            "perplexity_attempts_per_sec": ppl_rate,
            "perplexity_runner_source_sha256": manifest["perplexity_runner_source_sha256"],
            "perplexity_runner_source_file": manifest["perplexity_runner_source_file"],
        })
    return pd.DataFrame(records), {"runs": run_manifests, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--aux-csv", type=Path, action="append", required=True)
    parser.add_argument("--challenger-dir", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, underscores, and hyphens")
    challenger_dir = args.challenger_dir.resolve()
    if not challenger_dir.is_dir():
        raise FileNotFoundError(f"Challenger portfolio directory does not exist: {challenger_dir}")
    library_path = challenger_dir / "full_50k_library.parquet"
    manifest_path = challenger_dir / "assembly_manifest.json"
    if library_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite assembled submission library in {challenger_dir}")

    master, master_provenance = load_eligible_master()
    top_path = challenger_dir / "top100.parquet"
    portfolio_manifest_path = challenger_dir / "manifest.json"
    if not top_path.is_file() or not portfolio_manifest_path.is_file():
        raise FileNotFoundError("Challenger Top 100 and its manifest are required")
    portfolio_manifest = json.loads(portfolio_manifest_path.read_text())
    if sha256_file(top_path) != portfolio_manifest["outputs_sha256"]["top100.parquet"]:
        raise ValueError("Challenger Top 100 hash disagrees with its manifest")
    top100 = pd.read_parquet(top_path).sort_values("portfolio_rank", kind="stable").reset_index(drop=True)
    if len(top100) != 100 or top100["sequence"].nunique() != 100:
        raise ValueError("Challenger Top 100 must contain 100 unique sequences")
    if set(top100["primary_domain"]) != {"autoregressive", "evolution"}:
        raise ValueError("Top 100 must contain only the selected AR/evolution methods")
    if not all(
        set(str(value).split(";")) <= {"autoregressive", "evolution"}
        for value in top100["contributing_domains"]
    ):
        raise ValueError("Top 100 contains ancestry from an excluded baseline domain")
    import itertools
    import Levenshtein

    actual_internal_similarity = max(
        Levenshtein.ratio(left, right)
        for left, right in itertools.combinations(top100["sequence"], 2)
    )
    if actual_internal_similarity > 0.80:
        raise ValueError(f"Top-100 internal ratio exceeds 0.80: {actual_internal_similarity}")

    reference_path = ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    reference_rows, _, malformed = fasta_rows(reference_path)
    reference = {norm(row["sequence"]) for row in reference_rows}
    if malformed or len(reference_rows) != REFERENCE_COUNT or len(reference) != REFERENCE_COUNT:
        raise ValueError(f"Expected {REFERENCE_COUNT} valid unique reference peptides")
    max_reference_similarity = max(
        Levenshtein.ratio(sequence, reference_sequence)
        for sequence in top100["sequence"]
        for reference_sequence in reference
    )
    if max_reference_similarity > 0.80:
        raise ValueError(f"Top-100 reference similarity exceeds 0.80: {max_reference_similarity}")

    library_records = []
    seen = set()
    for _, row in master.iterrows():
        sequence = norm(row["sequence"])
        if not (8 <= len(sequence) <= 50) or not set(sequence) <= STANDARD_AA:
            raise ValueError(f"Invalid eligible master sequence: {sequence!r}")
        if sequence in reference:
            raise ValueError("An eligible AR/evolution sequence matches an official reference")
        if sequence in seen:
            raise ValueError("Eligible master contains duplicate sequences")
        seen.add(sequence)
        candidate_id = row.get("candidate_id")
        if pd.isna(candidate_id) or str(candidate_id).strip().lower() in {"", "nan", "none"}:
            candidate_id = f"seqsha_{hashlib.sha256(sequence.encode('ascii')).hexdigest()[:12]}"
        library_records.append({
            "library_index": len(library_records) + 1,
            "sequence_id": str(candidate_id),
            "sequence": sequence,
            "length": len(sequence),
            "source_tier": "scored_ar_evolution_candidate",
            "primary_domain": str(row["primary_domain"]),
            "contributing_domains": str(row["contributing_domains"]),
            "pred_amp_probability": float(row["pred_amp_probability"]),
            "pred_toxicity_risk": float(row["pred_toxicity_risk"]),
            "apex_mean_mic": float(row["apex_mean_mic"]),
            "evaluation_status": "role06_amp_safety_and_apex_scored",
        })

    if not set(top100["sequence"]).issubset(seen):
        raise ValueError("Top 100 must be a strict subset of eligible AR/evolution candidates")
    auxiliary, auxiliary_provenance = load_auxiliary(args.aux_csv, reference, seen)
    needed = LIBRARY_SIZE - len(library_records)
    if len(auxiliary) < needed:
        raise ValueError(
            f"Only {len(auxiliary)} unique non-reference ProGen2 fillers are available; "
            f"need {needed}. Generate another auxiliary batch before assembly."
        )

    model_dir = ROOT / "shared-evaluator/models"
    scores = score_sequences(auxiliary["sequence"].tolist(), model_dir)
    synthesis_results = [is_synthesizable(sequence) for sequence in auxiliary["sequence"]]
    auxiliary["synthesizable"] = [result[0] for result in synthesis_results]
    auxiliary["synthesis_reason"] = [result[1] for result in synthesis_results]
    auxiliary = pd.concat([auxiliary.reset_index(drop=True), scores], axis=1)
    auxiliary["evaluation_status"] = "role06_amp_safety_scored_apex_missing"
    synthesizable_pool = auxiliary[auxiliary["synthesizable"]].copy()
    if len(synthesizable_pool) < needed:
        raise ValueError(
            f"Only {len(synthesizable_pool)} unique ProGen2 fillers pass the synthesis filter; "
            f"need {needed}. Generate another auxiliary batch before assembly."
        )
    # Select only synthesis-filter-passing peptides, ranked by the Role 06 AMP/safety screen.
    auxiliary = synthesizable_pool.sort_values(
        ["library_selection_score", "sequence_id"],
        ascending=[False, True], kind="stable",
    ).head(needed)
    for _, row in auxiliary.iterrows():
        record = row.to_dict()
        record["library_index"] = len(library_records) + 1
        record["apex_mean_mic"] = None
        library_records.append(record)

    library = pd.DataFrame(library_records)
    sequences = library["sequence"].tolist()
    if len(library) != LIBRARY_SIZE or len(set(sequences)) != LIBRARY_SIZE:
        raise ValueError("Assembled library must contain exactly 50,000 unique sequences")
    if set(sequences) & reference:
        raise ValueError("Assembled library has an exact antibacterial reference overlap")
    if not set(top100["sequence"]).issubset(set(sequences)):
        raise ValueError("Top 100 is not a subset of the assembled library")
    library_synthesis = [is_synthesizable(sequence) for sequence in sequences]
    synthesis_failures = [reason for ok, reason in library_synthesis if not ok]
    if synthesis_failures:
        raise ValueError(
            f"Assembled library contains {len(synthesis_failures)} non-synthesizable sequences"
        )
    library_staging = library_path.with_name(f".{library_path.name}.staging")
    library.to_parquet(library_staging, index=False)
    library_staging.replace(library_path)
    manifest = {
        "run_id": args.run_id,
        "method": "scored_AR_evolution_candidates_plus_finetuned_ProGen2_fillers",
        "target_library_size": LIBRARY_SIZE,
        "actual_library_size": int(len(library)),
        "unique_sequences": int(library["sequence"].nunique()),
        "synthesis_filter_pass_count": int(len(library_synthesis) - len(synthesis_failures)),
        "top100_count": int(len(top100)),
        "top100_max_internal_levenshtein_ratio": actual_internal_similarity,
        "top100_max_reference_levenshtein_ratio": max_reference_similarity,
        "top100_is_subset": True,
        "exact_reference_overlap_count": 0,
        "reference_count": REFERENCE_COUNT,
        "reference_fasta_sha256": sha256_file(reference_path),
        "eligible_master_provenance": master_provenance,
        "challenger_top100_sha256": sha256_file(top_path),
        "challenger_manifest_sha256": sha256_file(portfolio_manifest_path),
        "auxiliary_provenance": auxiliary_provenance,
        "amp_model_sha256": sha256_file(model_dir / "amp_classifier_ensemble.joblib"),
        "hemolysis_model_sha256": sha256_file(model_dir / "empirical_hemolysis_ensemble.joblib"),
        "auxiliary_sequences_selected": int(needed),
        "unique_nonreference_auxiliary_pool": int(len(scores)),
        "synthesizable_unique_auxiliary_pool": int(len(synthesizable_pool)),
        "auxiliary_synthesizable_selected": int(auxiliary["synthesizable"].sum()),
        "auxiliary_selection_score_min": float(auxiliary["library_selection_score"].min()),
        "library_domain_counts": {str(k): int(v) for k, v in library["primary_domain"].value_counts().items()},
        "library_sha256": sha256_file(library_path),
        "scoring_limit": "Auxiliary MIC is unscored; AMP and safety fields are model predictions, not assay results.",
    }
    manifest_staging = manifest_path.with_name(f".{manifest_path.name}.staging")
    manifest_staging.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_staging.replace(manifest_path)
    print(json.dumps({
        "library": str(library_path),
        "rows": len(library),
        "unique": library["sequence"].nunique(),
        "auxiliary_fill": needed,
        "synthesizable_auxiliary": int(auxiliary["synthesizable"].sum()),
        "domain_counts": manifest["library_domain_counts"],
        "sha256": manifest["library_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
