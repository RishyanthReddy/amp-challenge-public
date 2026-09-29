"""
Role 03: VAE & Latent Models — Sequence Validation
Enforces official competition constraints on decoded peptides.
"""

from __future__ import annotations
import re

STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 50

def is_valid_sequence(sequence: str) -> tuple[bool, str]:
    seq = str(sequence or "").strip().upper()
    if not seq:
        return False, "empty_decode"
    non_canonical = set(seq) - STANDARD_AMINO_ACIDS
    if non_canonical:
        return False, f"invalid_residues_{''.join(sorted(non_canonical))}"
    if len(seq) < MIN_LENGTH:
        return False, f"too_short_{len(seq)}"
    if len(seq) > MAX_LENGTH:
        return False, f"too_long_{len(seq)}"
    return True, "ok"

def passes_biological_synthesizability(sequence: str) -> tuple[bool, str]:
    """
    Biological feasibility screen from HydrAMP (Szymczak et al. 2023, Methods):
    1. No cysteines (avoids disulfide scrambling in linear free peptides)
    2. No >= 3 positive residues in any 5-residue window (charge aggregation guard)
    3. No 3 identical residues in a row (homopolymer repeat guard)
    4. No 3 identical hydrophobic residues in a row (hydrophobic cluster guard)
    """
    seq = str(sequence or "").strip().upper()
    if "C" in seq:
        return False, "contains_cysteine"
    
    # 3 identical residues in a row
    if re.search(r"(.)\1\1", seq):
        return False, "homopolymer_triplet"
    
    # Positive cluster check: >= 3 positive (K, R, H) in any 5-window
    positive_set = set("KRH")
    for i in range(len(seq) - 4):
        window = seq[i:i+5]
        if sum(1 for aa in window if aa in positive_set) >= 3:
            return False, "positive_cluster_5win"
            
    # Hydrophobic triplet: 3 identical hydrophobic residues in a row
    for aa in "VILFWM":
        if aa * 3 in seq:
            return False, f"hydrophobic_triplet_{aa}"
            
    return True, "ok"
