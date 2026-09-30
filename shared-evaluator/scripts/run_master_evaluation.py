"""
Master Evaluation & Multi-Modal Integration Pipeline (Role 06)
Ingests all 13,425 candidates across 4 generation tracks:
- Autoregressive Models (outputs/ar_candidates.csv)
- VAE / HydrAMP Models (outputs/vae_candidates.csv)
- Sequence Diffusion Models (outputs/diffusion_candidates.csv)
- Evolutionary Search (outputs/evolution_candidates.csv)

Executes Stages 0 to 4:
Stage 0: Hard Validation (Alphabet, Length, 0 Ref Overlap)
Stage 1: Biophysical Descriptors (Charge, Moment, Boman, GRAVY, Instability)
Stage 2: ML Activity & Safety Predictors (P_AMP, sigma_AMP, R_tox, sigma_tox)
Stage 3: APEX Pathogen Ensemble MICs (11 clinical strains + mean/min MIC)
Stage 4: Multi-Domain Provenance Mapping and Cross-Domain Consensus

Exports:
- outputs/master_scored_candidates.csv
- shared-evaluator/reports/cross_domain_overlap.md
"""

import sys
import time
import math
import hashlib
import json
from pathlib import Path
from importlib.metadata import version as package_version
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(MAIN_ROOT / "data-engineering/src"))
from amp_data.core import fasta_rows, norm
from amp_data.synthesis_filter import is_synthesizable
from amp_data.ordering import sort_descending_portable

from evaluator.validator import validate_hard_rules
from evaluator.biophysical import compute_biophysical_properties
from evaluator.activity_safety_models import ActivitySafetyEvaluator
from evaluator.apex_scorer import ApexScorer

def main():
    print("=======================================================")
    print(" Running Master Multi-Modal Evaluation Pipeline (Role 06)")
    print(" Candidate Pool: current role outputs across 4 domains")
    print("=======================================================\n")

    t_start = time.time()

    # 1. Ingest candidates from all 4 domains
    files = {
        "autoregressive": MAIN_ROOT / "outputs/ar_candidates.csv",
        "vae_latent": MAIN_ROOT / "outputs/vae_candidates.csv",
        "diffusion": MAIN_ROOT / "outputs/diffusion_candidates.csv",
        "evolution": MAIN_ROOT / "outputs/evolution_candidates.csv",
    }

    all_dfs = []
    for domain, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path)
        required_columns = {"sequence", "max_reference_similarity", "passes_novelty_rule_le80"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"{domain} candidate file is missing required audit fields: {sorted(missing_columns)}")
        similarity = pd.to_numeric(df["max_reference_similarity"], errors="coerce")
        if not np.isfinite(similarity.to_numpy(dtype=float)).all() or df["passes_novelty_rule_le80"].isna().any():
            raise ValueError(f"{domain} candidates have missing or non-finite novelty audit fields")
        df["max_reference_similarity"] = similarity
        def parse_novelty_flag(value):
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
                return value.strip().lower() in {"true", "1"}
            raise ValueError(f"{domain} has invalid passes_novelty_rule_le80 value: {value!r}")
        df["passes_novelty_rule_le80"] = df["passes_novelty_rule_le80"].map(parse_novelty_flag)
        similarity_rule = df["max_reference_similarity"] <= 0.80
        if not np.array_equal(df["passes_novelty_rule_le80"].to_numpy(dtype=bool), similarity_rule.to_numpy(dtype=bool)):
            raise ValueError(f"{domain} novelty flags disagree with max_reference_similarity <= 0.80")
        df["ingested_domain"] = domain
        all_dfs.append(df)
        print(f"Loaded {len(df)} candidates from {domain} ({path.name})")

    df_raw = pd.concat(all_dfs, ignore_index=True)
    total_raw = len(df_raw)
    print(f"\nTotal raw ingested candidates: {total_raw}")

    # 2. Load official reference
    ref_path = MAIN_ROOT / "data-engineering/data/challenge/antibacterial.fasta"
    ref_rows, _, _ = fasta_rows(ref_path)
    ref_set = {norm(r["sequence"]) for r in ref_rows}
    print(f"Loaded {len(ref_set)} reference sequences.")

    # 3. Stages 0, 1, 2: Hard Validation, Biophysical Properties, ML Predictions
    print("\n[*] Running Stages 0, 1, 2: Hard Checks, Biophysical Profiling & ML Activity/Safety...")
    model_dir = ROOT / "models"
    ml_evaluator = ActivitySafetyEvaluator(model_dir=model_dir)

    scored_records = []
    t_stage1 = time.time()
    for idx, row in df_raw.iterrows():
        seq = str(row["sequence"]).strip().upper()
        cand_id = row.get("sequence_id", f"raw_cand_{idx:05d}")
        domain = row["ingested_domain"]
        model = row.get("model", "Unknown")

        # Stage 0: Hard Checks
        ok, errs = validate_hard_rules(seq, reference_set=ref_set)

        # Stage 1: Biophysical Properties
        props = compute_biophysical_properties(seq) if ok else {}

        # Stage 2: ML Activity & Safety Predictions
        ml_preds = ml_evaluator.predict(seq) if ok else {
            "pred_amp_probability": 0.0,
            "uncertainty_amp_std": 1.0,
            "pred_toxicity_risk": 1.0,
            "uncertainty_toxicity_std": 1.0,
        }

        # Synthesizability
        synth_ok, synth_reason = is_synthesizable(seq)

        scored_records.append({
            "candidate_id": cand_id,
            "sequence": seq,
            "domain": domain,
            "model": model,
            "is_valid": ok,
            "validation_errors": ";".join(errs),
            "synthesizable": synth_ok,
            "synthesis_flag": synth_reason,
            "max_reference_similarity": float(row["max_reference_similarity"]),
            "passes_novelty_rule_le80": bool(row["passes_novelty_rule_le80"]),
            **props,
            **ml_preds,
        })

        if (idx + 1) % 4000 == 0 or (idx + 1) == total_raw:
            print(f"  Processed {idx + 1}/{total_raw} sequences ({time.time()-t_stage1:.1f}s)")

    df_scored = pd.DataFrame(scored_records)

    # 4. Stage 4: Cross-Domain Deduplication and Provenance Aggregation
    print("\n[*] Running Stage 4: Cross-Domain Deduplication & Provenance Aggregation...")
    unique_pool = []
    domain_overlap_counts = {d: 0 for d in files.keys()}

    for seq, group in df_scored.groupby("sequence"):
        domains = sorted([str(d) for d in group["domain"].dropna().unique()])
        cand_ids = [str(c) for c in group["candidate_id"].dropna()]
        models = sorted([str(m) for m in group["model"].dropna().unique()])

        primary_rec = group.iloc[0].to_dict()
        primary_rec["primary_domain"] = domains[0]
        primary_rec["contributing_domains"] = ";".join(str(d) for d in domains)
        primary_rec["contributing_candidate_ids"] = ";".join(str(c) for c in cand_ids)
        primary_rec["contributing_models"] = ";".join(str(m) for m in models)
        primary_rec["domain_count"] = len(domains)

        for d in domains:
            domain_overlap_counts[d] += 1

        unique_pool.append(primary_rec)

    df_unique = pd.DataFrame(unique_pool)
    total_unique = len(df_unique)
    print(f"Total unique sequences retained: {total_unique}")

    # 5. Stage 3: APEX Multi-Strain Scoring for all candidates eligible for ranking
    print("\n[*] Running Stage 3: APEX Multi-Strain Scorer...")
    df_sorted = sort_descending_portable(df_unique, "pred_amp_probability").reset_index(drop=True)

    rankable_mask = (
        df_sorted["is_valid"].astype(bool)
        & df_sorted["synthesizable"].astype(bool)
        & df_sorted["passes_novelty_rule_le80"].astype(bool)
        & (df_sorted["max_reference_similarity"] <= 0.80)
    )
    rankable = df_sorted.loc[rankable_mask]
    top_seqs = rankable["sequence"].tolist()
    print(
        f"Scoring all {len(top_seqs)} valid, synthesizable candidates "
        "with the 11-pathogen APEX ensemble..."
    )
    if not top_seqs:
        raise RuntimeError("No candidates pass hard validation and synthesizability for APEX scoring")

    apex_scorer = ApexScorer()
    t_apex = time.time()
    df_apex_res = apex_scorer.score_batch(top_seqs, batch_size=1000)
    print(f"✓ APEX scoring complete in {time.time() - t_apex:.1f}s ({len(df_apex_res)} scored)")
    if df_apex_res.index.has_duplicates or set(df_apex_res.index.astype(str)) != set(top_seqs):
        missing = set(top_seqs) - set(df_apex_res.index.astype(str))
        extra = set(df_apex_res.index.astype(str)) - set(top_seqs)
        raise RuntimeError(
            f"APEX coverage mismatch: expected {len(top_seqs)}, got {len(df_apex_res)}; "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    # Merge APEX results
    df_final = df_sorted.merge(df_apex_res, left_on="sequence", right_index=True, how="left")
    print(f"✓ Merged APEX predictions into candidate master table.")
    if df_final.loc[rankable_mask, "apex_mean_mic"].isna().any():
        raise RuntimeError("At least one rankable candidate has missing APEX predictions after merge")

    # 6. Save outputs
    out_master_local = ROOT / "outputs/master_scored_candidates.csv"
    out_master_main = MAIN_ROOT / "outputs/master_scored_candidates.csv"
    df_final.to_csv(out_master_local, index=False)
    df_final.to_csv(out_master_main, index=False)
    print(f"\n✓ Saved Master Scored Candidate Table to:")
    print(f"  - {out_master_local}")
    print(f"  - {out_master_main}")

    source_hashes = {
        domain: hashlib.sha256(path.read_bytes()).hexdigest()
        for domain, path in files.items()
    }
    def hash_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    classifier_paths = {
        "amp_classifier_ensemble": model_dir / "amp_classifier_ensemble.joblib",
        "empirical_hemolysis_ensemble": model_dir / "empirical_hemolysis_ensemble.joblib",
    }
    if any(not path.is_file() for path in classifier_paths.values()):
        raise FileNotFoundError("A required current AMP or empirical-hemolysis model file is missing")
    apex_dir = MAIN_ROOT / "diffusion-models/apex"
    apex_files = [
        apex_dir / "APEX_predict.py",
        apex_dir / "APEX_models.py",
        apex_dir / "utils.py",
        apex_dir / "aaindex1.csv",
        *sorted((apex_dir / "APEX_pathogen_models").glob("*")),
    ]
    if any(not path.is_file() for path in apex_files):
        raise FileNotFoundError("APEX source, feature table, or pathogen model files are missing")
    evaluator_code_paths = [
        MAIN_ROOT / "data-engineering/src/amp_data/ordering.py",
        ROOT / "src/evaluator/validator.py",
        ROOT / "src/evaluator/biophysical.py",
        ROOT / "src/evaluator/activity_safety_models.py",
        ROOT / "src/evaluator/apex_scorer.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_candidate_rows": int(total_raw),
        "unique_sequences": int(total_unique),
        "valid_sequences": int(df_unique["is_valid"].sum()),
        "novelty_eligible_sequences": int((df_unique["max_reference_similarity"] <= 0.80).sum()),
        "synthesizable_novelty_eligible_sequences": int((df_unique["synthesizable"].astype(bool) & (df_unique["max_reference_similarity"] <= 0.80)).sum()),
        "rankable_sequences_scored_by_apex": int(len(top_seqs)),
        "apex_rows_returned": int(len(df_apex_res)),
        "apex_columns": list(df_apex_res.columns),
        "input_sha256": source_hashes,
        "classifier_model_sha256": {name: hash_path(path) for name, path in classifier_paths.items()},
        "apex_asset_sha256": {path.relative_to(MAIN_ROOT).as_posix(): hash_path(path) for path in apex_files},
        "evaluator_code_sha256": {path.relative_to(MAIN_ROOT).as_posix(): hash_path(path) for path in evaluator_code_paths},
        "training_data_summary_sha256": hash_path(ROOT / "reports/training_data_summary.json"),
        "runtime": {
            package: package_version(package)
            for package in ("numpy", "pandas", "scikit-learn", "joblib")
        },
        "output_sha256": hashlib.sha256(out_master_main.read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    manifest_path = ROOT / "reports/master_evaluation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"✓ Saved evaluation manifest to {manifest_path}")

    # 7. Generate Cross-Domain Overlap Report
    multi_domain = df_final[df_final["domain_count"] > 1]
    exclusive_counts = {
        domain: int((df_final["contributing_domains"] == domain).sum())
        for domain in files
    }
    ref_matches = int(df_unique["validation_errors"].str.contains("exact_match_to_official_reference").sum())
    input_counts = {domain: len(pd.read_csv(path)) for domain, path in files.items()}
    overlap_md = f"""# Role 06: Cross-Domain Overlap & Candidate Reservoir Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Ingested Candidates:** {total_raw}
**Total Unique Candidates:** {total_unique}
**Cross-Domain Consensus Sequences:** {len(multi_domain)}

---

## 1. Domain Contribution Breakdown

| Domain | Ingested Candidates | Unique Sequences Present | Exclusive Sequences |
| :--- | :--- | :--- | :--- |
| **Autoregressive LMs** | {input_counts['autoregressive']} | {domain_overlap_counts['autoregressive']} | {exclusive_counts['autoregressive']} |
| **VAE / HydrAMP** | {input_counts['vae_latent']} | {domain_overlap_counts['vae_latent']} | {exclusive_counts['vae_latent']} |
| **Sequence Diffusion** | {input_counts['diffusion']} | {domain_overlap_counts['diffusion']} | {exclusive_counts['diffusion']} |
| **Evolutionary Search** | {input_counts['evolution']} | {domain_overlap_counts['evolution']} | {exclusive_counts['evolution']} |
| **Total Reservoir** | **{total_raw}** | **{total_unique} (Total Unique)** | **{sum(exclusive_counts.values())} (Total Exclusive)** |

---

## 2. Multi-Domain Consensus Sequences

A total of **{len(multi_domain)} sequences** were generated in multiple distinct domains. This overlap records provenance only and does not establish biological activity.

---

## 3. APEX Pathogen Scores

The 11-pathogen ensemble scored **{len(df_apex_res)} / {len(top_seqs)}** valid, synthesizable
candidates. Mean predicted MIC: {df_final['apex_mean_mic'].dropna().mean():.2f} uM;
lowest predicted MIC: {df_final['apex_min_mic'].dropna().min():.2f} uM. These are model
predictions, not measured MIC results. Exact reference matches found: **{ref_matches}**.
"""

    rep_overlap = ROOT / "reports/cross_domain_overlap.md"
    with open(rep_overlap, "w") as f:
        f.write(overlap_md)
    print(f"✓ Saved cross-domain overlap report to {rep_overlap}")

    total_time = time.time() - t_start
    print(f"\n=======================================================")
    print(f" Master Evaluation Pipeline Finished in {total_time:.1f}s")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
