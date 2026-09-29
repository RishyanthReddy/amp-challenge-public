"""AMP Challenge 2027 baseline generator (HydrAMP).

Produces a default-settings 50,000-sequence library and a ranked top-100 using the
published HydrAMP conditional VAE. Output goes to ``generate_broad_spectrum/``:

    generate_broad_spectrum/library.fasta   full 50,000-sequence library
    generate_broad_spectrum/top.fasta       top-100 ranked candidates

HydrAMP is a competition baseline (excluded from rankings), so generation uses
HydrAMP's default AMP settings; only the challenge's hard sequence rules are
enforced on top (20 canonical residues, length 8-50, uniqueness, no overlap with
the antibacterial reference, and <80% top-100 similarity to it).
"""

from __future__ import annotations

import os

# Must be set before importing HydrAMP (pulls TensorFlow + matplotlib at import time).
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import sys
from pathlib import Path

import Levenshtein

CATEGORY = "generate_broad_spectrum"
MODEL_PATH = "checkpoint/model"
DECOMPOSER_PATH = "checkpoint/pca_decomposer.joblib"
ANTIBACTERIAL_FASTA = "data/antibacterial.fasta"

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50
SIMILARITY_THRESHOLD = 0.8


def _read_fasta_sequences(path: Path) -> list[str]:
    sequences, parts = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line.upper())
    if parts:
        sequences.append("".join(parts))
    return sequences


def _write_fasta(sequences: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


def _is_valid(seq: str) -> bool:
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH and not (set(seq) - STANDARD_AMINO_ACIDS)


def generate_library(
    generator,
    n_sequences: int,
    seed: int,
    references: set[str],
) -> dict[str, float]:
    """Accumulate `n_sequences` unique valid sequences mapped to their HydrAMP score.

    HydrAMP decoding is deterministic per seed (softmax path), so each round uses a
    fresh, deterministic seed and accepted sequences are kept in insertion order, so
    the whole procedure is reproducible.
    """
    collected: dict[str, float] = {}
    round_seed = seed
    max_rounds = 1000
    round_idx = 0
    while len(collected) < n_sequences and round_idx < max_rounds:
        need = n_sequences - len(collected)
        batch = generator.unconstrained_generation(
            mode="amp",
            n_target=need,
            seed=round_seed,
            filter_out=True,
            properties=True,
            n_attempts=1,
        )
        # `batch` is a list of per-sequence dicts. HydrAMP ranks AMP candidates by `mic`
        # (probability of low MIC; higher is better).
        for item in batch:
            seq = str(item["sequence"])
            if seq in collected or seq in references or not _is_valid(seq):
                continue
            collected[seq] = float(item["mic"])
        print(f"  seed {round_seed}: {len(collected)}/{n_sequences} collected")
        round_seed += 1
        round_idx += 1
    if len(collected) < n_sequences:
        raise RuntimeError(
            f"Generation failed: collected only {len(collected)}/{n_sequences} valid sequences "
            f"after {round_idx} rounds. Target count not reached."
        )
    return dict(list(collected.items())[:n_sequences])


def _passes_biological_filters(seq: str) -> bool:
    """HydrAMP's stringent synthesizability criteria (Szymczak et al. 2023, Methods):
    no cysteines, no >=3 positive residues in any 5-window, no 3 identical residues in a
    row, and no 3 identical hydrophobic residues in a row."""
    from amp.inference.filtering import (
        check_for_cysteins,
        check_sequence_for_hydrophobic_clusters,
        check_sequence_for_positive_clusters,
        check_sequence_for_repetitive_clusters,
    )
    return (
        check_for_cysteins(seq)
        and check_sequence_for_positive_clusters(seq)
        and check_sequence_for_repetitive_clusters(seq)
        and check_sequence_for_hydrophobic_clusters(seq)
    )


def select_top(library: dict[str, float], top_k: int, references: set[str]) -> list[str]:
    """Top-`top_k` library sequences ranked by HydrAMP's MIC score, restricted to
    candidates that pass HydrAMP's biological synthesizability filters and stay under 80%
    similarity to every antibacterial reference.

    This mirrors the paper's biological-filtering + ranking preselection stage. The
    paper's further classifier-consensus and molecular-dynamics steps are intentionally
    not reproduced: they require external tools and GPU-days of simulation and are
    non-deterministic, which is incompatible with a single-command, reproducible run.
    """
    ranked = sorted(library, key=lambda s: library[s], reverse=True)
    top: list[str] = []
    for seq in ranked:
        if not _passes_biological_filters(seq):
            continue
        if all(Levenshtein.ratio(seq, ref) <= SIMILARITY_THRESHOLD for ref in references):
            top.append(seq)
            if len(top) == top_k:
                break
    if len(top) < top_k:
        raise RuntimeError(
            f"Only {len(top)} of {top_k} sequences passed the biological + <"
            f"{SIMILARITY_THRESHOLD} similarity filters; increase --n-sequences."
        )
    return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--decomposer-path", default=DECOMPOSER_PATH)
    parser.add_argument("--antibacterial-fasta", default=ANTIBACTERIAL_FASTA)
    args = parser.parse_args()

    if args.n_sequences <= 0:
        parser.error("--n-sequences must be greater than zero")
    if args.top_k <= 0 or args.top_k > args.n_sequences:
        parser.error("--top-k must be between 1 and --n-sequences")

    for path in (args.model_path, args.decomposer_path, args.antibacterial_fasta):
        if not Path(path).exists():
            sys.exit(f"ERROR: required path not found (cwd: {os.getcwd()}): {path}")

    from amp.inference.inference import HydrAMPGenerator

    references = set(_read_fasta_sequences(Path(args.antibacterial_fasta)))
    generator = HydrAMPGenerator(
        model_path=args.model_path,
        decomposer_path=args.decomposer_path,
        softmax=True,
    )

    out_dir = Path(CATEGORY)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_sequences} sequences (seed={args.seed})")
    library = generate_library(generator, args.n_sequences, args.seed, references)
    sequences = list(library)
    _write_fasta(sequences, out_dir / "library.fasta")
    print(f"Wrote {len(sequences)} sequences -> {out_dir / 'library.fasta'}")

    top = select_top(library, args.top_k, references)
    _write_fasta(top, out_dir / "top.fasta")
    print(f"Wrote top {len(top)} sequences -> {out_dir / 'top.fasta'}")
