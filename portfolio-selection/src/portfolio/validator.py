"""
Official Challenge Validator Wrapper (Role 07)
Strictly mirrors and extends the official verification logic from verify_submission.py:
1. Library verification: exactly 50,000 unique sequences, 20 canonical AAs, 8 <= L <= 50.
2. Reference overlap: exactly 0 exact matches to antibacterial.fasta.
3. Top list verification: exactly 100 sequences, 100% strict subset of library.fasta, 0 duplicates.
4. Novelty rule: Levenshtein ratio <= 0.80 against all reference sequences.
5. Header syntax & cleanliness: strictly valid rank, domain, and candidate ID with zero 'nan' values.
"""

import re
from pathlib import Path
from typing import Tuple, List, Set
import Levenshtein

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50
TOP_SIZE = 100
LIBRARY_SIZE = 50_000

HEADER_REGEX = re.compile(r"^rank_\d{3}_(diffusion|autoregressive|evolution|vae_latent)_[a-zA-Z0-9_-]+$")

def read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    """Parse FASTA into headers and sequences."""
    headers, sequences = [], []
    header, seq_parts = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(seq_parts))
            header, seq_parts = line[1:], []
        else:
            seq_parts.append(line.upper())
    if header is not None:
        headers.append(header)
        sequences.append("".join(seq_parts))
    return headers, sequences

def verify_library_sequences(fasta_path: Path, expected_size: int = LIBRARY_SIZE) -> Tuple[bool, List[str], Set[str]]:
    """Verify full library constraints."""
    headers, sequences = read_fasta(fasta_path)
    errors: List[str] = []

    if not sequences:
        errors.append("FASTA file is empty — no sequences found.")
    elif len(sequences) != expected_size:
        errors.append(f"Expected {expected_size} sequences, got {len(sequences)}.")

    seen: Set[str] = set()
    for i, (header, seq) in enumerate(zip(headers, sequences), start=1):
        if not header.strip():
            errors.append(f"Record {i}: missing header.")
        if not seq:
            errors.append(f"Record {i} ('{header}'): empty sequence.")
            continue
        invalid = set(seq) - STANDARD_AMINO_ACIDS
        if invalid:
            errors.append(f"Record {i} ('{header}'): invalid characters {sorted(invalid)}.")
        if len(seq) < MIN_LENGTH:
            errors.append(f"Record {i} ('{header}'): sequence too short ({len(seq)} < {MIN_LENGTH}).")
        if len(seq) > MAX_LENGTH:
            errors.append(f"Record {i} ('{header}'): sequence too long ({len(seq)} > {MAX_LENGTH}).")
        if seq in seen:
            errors.append(f"Record {i} ('{header}'): duplicate sequence.")
        seen.add(seq)

    return len(errors) == 0, errors, seen

def verify_no_overlap(full_sequences: Set[str], reference_sequences: Set[str]) -> Tuple[bool, List[str]]:
    """Verify zero exact matches to reference sequences."""
    overlap = full_sequences & reference_sequences
    errors = []
    if overlap:
        errors.append(f"Overlap check failed: {len(overlap)} sequence(s) found in reference library.")
    return len(errors) == 0, errors

def verify_top_list(top_fasta_path: Path, full_sequences: Set[str], expected_top: int = TOP_SIZE) -> Tuple[bool, List[str], List[str]]:
    """Verify top list: size == 100, strict subset of library, 0 duplicates, and valid header syntax."""
    headers, top_sequences = read_fasta(top_fasta_path)
    errors: List[str] = []

    if len(top_sequences) != expected_top:
        errors.append(f"Expected {expected_top} sequences in top list, got {len(top_sequences)}.")

    seen: Set[str] = set()
    for i, (header, seq) in enumerate(zip(headers, top_sequences), start=1):
        # Header syntax check
        if "nan" in header.lower():
            errors.append(f"Record {i}: header contains 'nan': '{header}'")
        if not HEADER_REGEX.match(header):
            errors.append(f"Record {i}: malformed header syntax: '{header}'")

        # Subset check
        if seq not in full_sequences:
            errors.append(f"Record {i}: sequence '{seq[:10]}...' not found in full library (SUBSET VIOLATION).")
        if seq in seen:
            errors.append(f"Record {i}: duplicate sequence in top list.")
        seen.add(seq)

    return len(errors) == 0, errors, top_sequences

def verify_max_similarity(top_sequences: List[str], reference_sequences: Set[str], threshold: float = 0.80) -> Tuple[bool, List[str]]:
    """Verify that every sequence in top list has Levenshtein ratio <= threshold vs all references."""
    errors = []
    for seq in top_sequences:
        for ref in reference_sequences:
            sim = Levenshtein.ratio(seq, ref)
            if sim > threshold:
                errors.append(f"Novelty check failed: sequence '{seq}' has similarity {sim:.4f} > {threshold} vs reference '{ref}'.")
                break
    return len(errors) == 0, errors
