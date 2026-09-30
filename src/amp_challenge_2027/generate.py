"""
Validated artifact exporter for AMP Challenge 2027
Entry point command: uv run export_submission
Generates:
- generate_broad_spectrum/library.fasta (50,000 sequences)
- generate_broad_spectrum/top.fasta (100 candidate sequences)
"""

import argparse
import itertools
import sys
import shutil
import hashlib
import re
from pathlib import Path
import pandas as pd
import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
REFERENCE_SEQUENCE_COUNT = 39_448

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _clean_id(value, fallback: str) -> str:
    """Return a stable FASTA-safe identifier, replacing missing-value sentinels."""
    if pd.isna(value):
        return fallback
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return fallback
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not value or "nan" in value.lower():
        return fallback
    return value


def _validated_sequence(value, source: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{source} contains a missing sequence")
    sequence = "".join(str(value).split()).upper()
    if not 8 <= len(sequence) <= 50:
        raise ValueError(f"{source} has invalid peptide length {len(sequence)}")
    invalid = set(sequence) - STANDARD_AMINO_ACIDS
    if invalid:
        raise ValueError(f"{source} contains noncanonical residues: {sorted(invalid)}")
    return sequence


def _read_fasta_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    has_header = False
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if has_header:
                if not current:
                    raise ValueError(f"Empty FASTA record before {path}:{line_number}")
                sequences.append("".join(current))
            has_header = True
            current = []
        else:
            if not has_header:
                raise ValueError(f"Sequence before first FASTA header at {path}:{line_number}")
            current.append("".join(line.split()).upper())
    if has_header:
        if not current:
            raise ValueError(f"Final FASTA record is empty: {path}")
        sequences.append("".join(current))
    return sequences

def main(
    top100_path: Path | None = None,
    lib_path: Path | None = None,
    out_dir: Path | None = None,
    ref_path: Path | None = None,
):
    print("=================================================================")
    print(" AMP Challenge 2027: Validated Submission Export")
    print("=================================================================\n")

    default_output = out_dir is None
    top100_path = top100_path or ROOT / "portfolio-selection/outputs/portfolio_top100.parquet"
    lib_path = lib_path or ROOT / "portfolio-selection/outputs/full_50k_library.parquet"
    ref_path = ref_path or ROOT / "data-engineering/data/challenge/antibacterial.fasta"

    if not top100_path.is_file() or not lib_path.is_file():
        raise FileNotFoundError(
            f"Missing assembled portfolio files. Expected:\n  {top100_path}\n  {lib_path}"
        )
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Required antibacterial reference FASTA is missing: {ref_path}. "
            "Novelty validation cannot be skipped."
        )

    df_top100 = pd.read_parquet(top100_path)
    df_lib = pd.read_parquet(lib_path)

    required_lib = {"library_index", "sequence_id", "sequence"}
    required_top = {"portfolio_rank", "sequence", "primary_domain", "candidate_id", "contributing_candidate_ids"}
    missing_lib = required_lib - set(df_lib.columns)
    missing_top = required_top - set(df_top100.columns)
    if missing_lib or missing_top:
        raise ValueError(
            f"Portfolio input schema is incomplete: library missing {sorted(missing_lib)}, "
            f"top-100 missing {sorted(missing_top)}"
        )
    if len(df_lib) != 50_000 or len(df_top100) != 100:
        raise ValueError(
            f"Expected 50,000 library and 100 ranked candidates; found "
            f"{len(df_lib)} and {len(df_top100)}"
        )

    df_lib = df_lib.sort_values("library_index", kind="stable").reset_index(drop=True)
    df_top100 = df_top100.sort_values("portfolio_rank", kind="stable").reset_index(drop=True)
    expected_ranks = list(range(1, 101))
    if df_top100["portfolio_rank"].tolist() != expected_ranks:
        raise ValueError("Top-100 ranks must be exactly 1 through 100 in order")
    if df_lib["library_index"].tolist() != list(range(1, 50_001)):
        raise ValueError("Library indices must be exactly 1 through 50,000")

    lib_records = []
    for position, row in df_lib.iterrows():
        seq = _validated_sequence(row["sequence"], f"library row {position + 1}")
        lib_records.append((row, seq))
    lib_seqs = [seq for _, seq in lib_records]
    if len(set(lib_seqs)) != 50_000:
        raise ValueError("Library sequences must be unique")

    top_records = []
    for _, row in df_top100.iterrows():
        rank = int(row["portfolio_rank"])
        seq = _validated_sequence(row["sequence"], f"top rank {rank}")
        top_records.append((row, seq))
    top_seqs = [seq for _, seq in top_records]
    if len(set(top_seqs)) != 100:
        raise ValueError("Top-100 sequences must be unique")
    if not set(top_seqs).issubset(set(lib_seqs)):
        raise ValueError("Top 100 must be a strict subset of the 50,000-sequence library")

    ref_records = _read_fasta_sequences(ref_path)
    if len(ref_records) != REFERENCE_SEQUENCE_COUNT:
        raise ValueError(
            f"Expected {REFERENCE_SEQUENCE_COUNT} reference records, found {len(ref_records)}"
        )
    ref_seqs = set(ref_records)
    if not ref_seqs:
        raise ValueError(f"Antibacterial reference FASTA contains no sequences: {ref_path}")
    overlap = set(lib_seqs) & ref_seqs
    if overlap:
        raise ValueError(f"Found {len(overlap)} exact matches in antibacterial reference")
    max_sim = max(Levenshtein.ratio(seq, ref) for seq in top_seqs for ref in ref_seqs)
    if max_sim > 0.80:
        raise ValueError(f"Top candidate similarity {max_sim:.6f} exceeds 0.80 threshold")
    max_internal_sim = max(
        Levenshtein.ratio(left, right)
        for left, right in itertools.combinations(top_seqs, 2)
    )
    if max_internal_sim > 0.80:
        raise ValueError(f"Top-100 internal similarity {max_internal_sim:.6f} exceeds 0.80 threshold")

    out_dir = out_dir or ROOT / "generate_broad_spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)

    lib_fasta = out_dir / "library.fasta"
    top_fasta = out_dir / "top.fasta"

    print(f"[1/4] Writing 50,000 library sequences to {lib_fasta}...")
    with open(lib_fasta, "w") as f:
        for row, seq in lib_records:
            idx = int(row["library_index"])
            seq_digest = hashlib.sha256(seq.encode("ascii")).hexdigest()[:12]
            candidate_id = _clean_id(row["sequence_id"], f"seqsha_{seq_digest}")
            f.write(f">seq_{idx:05d}_{candidate_id}\n{seq}\n")

    print(f"[2/4] Writing 100 ranked top candidates to {top_fasta}...")
    with open(top_fasta, "w") as f:
        for row, seq in top_records:
            rank = int(row["portfolio_rank"])
            domain = _clean_id(row["primary_domain"], "")
            if domain not in {"diffusion", "autoregressive", "evolution", "vae_latent"}:
                raise ValueError(f"Unexpected candidate domain at rank {rank}: {domain!r}")
            cand_id = _clean_id(row["candidate_id"], "")
            if not cand_id:
                cand_id = _clean_id(row["contributing_candidate_ids"], "")
            if not cand_id:
                cand_id = f"seqsha_{hashlib.sha256(seq.encode('ascii')).hexdigest()[:12]}"
            f.write(f">rank_{rank:03d}_{domain}_{cand_id}\n{seq}\n")

    # Mirror only the canonical run to generate/ for ENTRY_POINT="generate".
    if default_output:
        compat_dir = ROOT / "generate"
        compat_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lib_fasta, compat_dir / "library.fasta")
        shutil.copyfile(top_fasta, compat_dir / "top.fasta")

    print("[3/4] Validating generated FASTA invariants...")
    print("      FASTA counts, uniqueness, alphabet, lengths, subset, and reference novelty: PASS")
    print(f"      Max top-to-reference similarity: {max_sim:.4f} (<= 0.80)")
    print(f"      Max internal Top-100 similarity: {max_internal_sim:.4f} (<= 0.80)")

    print("[4/4] Computing cryptographic checksums...")
    lib_sha = sha256_file(lib_fasta)
    top_sha = sha256_file(top_fasta)
    print(f"      library.fasta SHA-256: {lib_sha}")
    print(f"      top.fasta     SHA-256: {top_sha}")

    print("\n✓ SUCCESS: Generation completed. All challenge invariants verified 100%!")

def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-parquet", type=Path)
    parser.add_argument("--library-parquet", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-fasta", type=Path)
    args = parser.parse_args()
    main(args.top_parquet, args.library_parquet, args.output_dir, args.reference_fasta)


if __name__ == "__main__":
    cli()
