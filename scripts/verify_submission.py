"""
Local and CI submission verification script for AMP Challenge 2027.
Validates:
1. Generation execution via entry point
2. FASTA formatting, length, canonical alphabet, uniqueness
3. Strict subset inclusion
4. Antibacterial reference non-overlap
5. Levenshtein novelty threshold (<= 0.80)
6. Generation replay and bit-for-bit reproducibility
"""

import subprocess
import argparse
import itertools
import re
from pathlib import Path
import Levenshtein

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50
TOP_SIZE = 100
LIBRARY_SIZE = 50_000
REFERENCE_SEQUENCE_COUNT = 39_448

TOP_HEADER = re.compile(
    r"^rank_\d{3}_(?:diffusion|autoregressive|evolution|vae_latent)_[A-Za-z0-9_-]+$"
)


def _read_fasta(path: Path) -> tuple[list[str], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required FASTA file does not exist: {path}")
    headers, sequences = [], []
    header, seq_parts = None, []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                if not seq_parts:
                    raise ValueError(f"Empty sequence for FASTA record {header!r} in {path}")
                headers.append(header)
                sequences.append("".join("".join(part.split()) for part in seq_parts).upper())
            header, seq_parts = line[1:], []
            if not header:
                raise ValueError(f"Empty FASTA header at {path}:{line_number}")
        else:
            if header is None:
                raise ValueError(f"Sequence text before first FASTA header at {path}:{line_number}")
            seq_parts.append(line)
    if header is not None:
        if not seq_parts:
            raise ValueError(f"Empty sequence for FASTA record {header!r} in {path}")
        headers.append(header)
        sequences.append("".join("".join(part.split()) for part in seq_parts).upper())
    return headers, sequences

def verify_submission(
    root_dir: Path,
    ref_path: Path | None = None,
    replay: bool = True,
    library_path: Path | None = None,
    top_path: Path | None = None,
):
    print("=======================================================")
    print(" AMP Challenge 2027: Comprehensive Submission Verifier")
    print("=======================================================\n")

    root_dir = root_dir.resolve()
    if ref_path is None:
        ref_path = root_dir / "data-engineering/data/challenge/antibacterial.fasta"
    else:
        ref_path = Path(ref_path)
        if not ref_path.is_absolute():
            ref_path = root_dir / ref_path
        ref_path = ref_path.resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Required antibacterial reference FASTA is missing: {ref_path}. "
            "Reference novelty checks cannot be skipped."
        )

    custom_paths = library_path is not None or top_path is not None
    if custom_paths and replay:
        raise ValueError("--replay applies only to the canonical generated FASTAs; use --no-replay for custom artifacts")
    lib_path = library_path
    if lib_path is not None and not lib_path.is_absolute():
        lib_path = root_dir / lib_path
    if top_path is not None and not top_path.is_absolute():
        top_path = root_dir / top_path
    if lib_path is None:
        lib_path = root_dir / "generate_broad_spectrum/library.fasta"
    if top_path is None:
        top_path = root_dir / "generate_broad_spectrum/top.fasta"

    if not custom_paths and not lib_path.exists():
        lib_path = root_dir / "generate/library.fasta"
        top_path = root_dir / "generate/top.fasta"

    print(f"[1] Auditing library at: {lib_path}")
    print(f"[2] Auditing top list at: {top_path}")

    lib_headers, lib_seqs = _read_fasta(lib_path)
    top_headers, top_seqs = _read_fasta(top_path)

    # 1. Size checks
    if len(lib_headers) != LIBRARY_SIZE or len(lib_seqs) != LIBRARY_SIZE:
        raise ValueError(f"Expected {LIBRARY_SIZE} library records, got {len(lib_seqs)}")
    if len(top_headers) != TOP_SIZE or len(top_seqs) != TOP_SIZE:
        raise ValueError(f"Expected {TOP_SIZE} top records, got {len(top_seqs)}")
    if len(set(lib_headers)) != LIBRARY_SIZE:
        raise ValueError("Library FASTA headers must be unique")
    if len(set(top_headers)) != TOP_SIZE:
        raise ValueError("Top FASTA headers must be unique")

    # 2. Uniqueness
    if len(set(lib_seqs)) != LIBRARY_SIZE:
        raise ValueError("Duplicates detected in library.fasta")
    if len(set(top_seqs)) != TOP_SIZE:
        raise ValueError("Duplicates detected in top.fasta")

    # 3. Canonical residues and length constraints
    for i, s in enumerate(lib_seqs):
        diff = set(s) - STANDARD_AMINO_ACIDS
        if diff:
            raise ValueError(f"Library seq {i} contains invalid amino acids: {sorted(diff)}")
        if not MIN_LENGTH <= len(s) <= MAX_LENGTH:
            raise ValueError(f"Library seq {i} length {len(s)} outside [{MIN_LENGTH}, {MAX_LENGTH}]")

    for i, s in enumerate(top_seqs):
        diff = set(s) - STANDARD_AMINO_ACIDS
        if diff:
            raise ValueError(f"Top seq {i} contains invalid amino acids: {sorted(diff)}")
        if not MIN_LENGTH <= len(s) <= MAX_LENGTH:
            raise ValueError(f"Top seq {i} length {len(s)} outside [{MIN_LENGTH}, {MAX_LENGTH}]")

    # 4. Strict subset
    lib_set = set(lib_seqs)
    for s in top_seqs:
        if s not in lib_set:
            raise ValueError(f"Top candidate {s} not found in library.fasta")

    # 5. Header checks
    for i, header in enumerate(lib_headers, start=1):
        if "nan" in header.lower():
            raise ValueError(f"Library header {i} contains a missing-value sentinel: {header}")
    for i, h in enumerate(top_headers, start=1):
        if "nan" in h.lower():
            raise ValueError(f"Top header {i} contains a missing-value sentinel: {h}")
        match = TOP_HEADER.fullmatch(h)
        if match is None:
            raise ValueError(f"Malformed top header {i}: {h}")
        if int(h[5:8]) != i:
            raise ValueError(f"Top header rank is out of order at record {i}: {h}")

    # 6. Reference overlap and similarity
    _, ref_seqs = _read_fasta(ref_path)
    ref_set = set(ref_seqs)
    if not ref_set:
        raise ValueError(f"Reference FASTA contains no records: {ref_path}")
    if len(ref_seqs) != REFERENCE_SEQUENCE_COUNT or len(ref_set) != REFERENCE_SEQUENCE_COUNT:
        raise ValueError(
            f"Expected {REFERENCE_SEQUENCE_COUNT} unique reference records; "
            f"found {len(ref_seqs)} records and {len(ref_set)} unique sequences"
        )
    overlap = lib_set & ref_set
    if overlap:
        raise ValueError(f"{len(overlap)} sequences overlap with reference FASTA")

    max_sim = 0.0
    for t in top_seqs:
        for r in ref_set:
            sim = Levenshtein.ratio(t, r)
            if sim > max_sim:
                max_sim = sim
            if sim > 0.80:
                raise ValueError(f"Novelty violation: candidate similarity {sim:.4f} > 0.80")
    print(f"    v Novelty check passed: max similarity to reference is {max_sim:.4f} <= 0.80")

    max_internal = max(
        Levenshtein.ratio(left, right)
        for left, right in itertools.combinations(top_seqs, 2)
    )
    if max_internal > 0.80:
        raise ValueError(f"Top-100 internal similarity {max_internal:.4f} exceeds 0.80")
    print(f"    v Internal Top-100 similarity passed: max is {max_internal:.4f} <= 0.80")

    # 7. Reproducibility replay check
    if replay:
        print("[3] Replaying generation to verify bit-for-bit reproducibility...")
        lib_before = lib_path.read_bytes()
        top_before = top_path.read_bytes()

        cmd = ["uv", "run", "generate_broad_spectrum"]
        res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Replay failed:\n{res.stderr}")

        if lib_path.read_bytes() != lib_before:
            raise ValueError("Reproducibility error: library.fasta differs across runs")
        if top_path.read_bytes() != top_before:
            raise ValueError("Reproducibility error: top.fasta differs across runs")
        print("    v Reproducibility check passed: identical bit-for-bit output across consecutive runs.")

    print("\n=======================================================")
    print(" ALL SUBMISSION INVARIANTS RIGOROUSLY VERIFIED (PASS)")
    print("=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default=".", type=Path)
    parser.add_argument("--ref", type=Path, default=None)
    parser.add_argument("--library-fasta", type=Path, default=None)
    parser.add_argument("--top-fasta", type=Path, default=None)
    parser.add_argument("--no-replay", dest="replay", action="store_false")
    args = parser.parse_args()
    verify_submission(
        args.dir,
        args.ref,
        replay=args.replay,
        library_path=args.library_fasta,
        top_path=args.top_fasta,
    )
