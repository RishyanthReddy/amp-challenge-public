"""
Mutation Operators for Evolutionary AMP Search (Role 05)
Implements:
- Point substitution
- Insertion (guarded by MAX_LEN = 50)
- Deletion (guarded by MIN_LEN = 8)
- Adaptive composite mutation
Guarantees 100% 20 canonical amino acids and strict [8, 50] residue length bounds.
"""

from typing import Tuple, Optional
import numpy as np

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_SET = set(STANDARD_AA)
MIN_LEN = 8
MAX_LEN = 50


def _validated_parent(seq: str) -> str:
    if not isinstance(seq, str):
        raise TypeError("Mutation parent must be a string")
    seq = seq.strip().upper()
    if not MIN_LEN <= len(seq) <= MAX_LEN:
        raise ValueError(f"Mutation parent length must be in [{MIN_LEN}, {MAX_LEN}]")
    invalid = set(seq) - STANDARD_AA_SET
    if invalid:
        raise ValueError(f"Mutation parent contains noncanonical residues: {sorted(invalid)}")
    return seq

def point_substitution(seq: str, rng: np.random.Generator) -> Tuple[str, str, int, str]:
    """Substitute a single amino acid in seq with another canonical amino acid."""
    seq = _validated_parent(seq)
    pos = int(rng.integers(0, len(seq)))
    old_aa = seq[pos]
    candidates = [aa for aa in STANDARD_AA if aa != old_aa]
    new_aa = candidates[int(rng.integers(0, len(candidates)))]
    new_seq = seq[:pos] + new_aa + seq[pos + 1:]
    delta = f"{old_aa}{pos}{new_aa}"
    return new_seq, "sub", pos, delta

def insertion(seq: str, rng: np.random.Generator) -> Tuple[str, str, int, str]:
    """Insert a canonical amino acid. Guarded against len(seq) >= MAX_LEN."""
    seq = _validated_parent(seq)
    if len(seq) >= MAX_LEN:
        # Fall back to substitution if max length reached
        return point_substitution(seq, rng)
    pos = int(rng.integers(0, len(seq) + 1))
    new_aa = STANDARD_AA[int(rng.integers(0, len(STANDARD_AA)))]
    new_seq = seq[:pos] + new_aa + seq[pos:]
    delta = f"ins_{pos}_{new_aa}"
    return new_seq, "ins", pos, delta

def deletion(seq: str, rng: np.random.Generator) -> Tuple[str, str, int, str]:
    """Delete a single residue. Guarded against len(seq) <= MIN_LEN."""
    seq = _validated_parent(seq)
    if len(seq) <= MIN_LEN:
        # Fall back to substitution if min length reached
        return point_substitution(seq, rng)
    pos = int(rng.integers(0, len(seq)))
    old_aa = seq[pos]
    new_seq = seq[:pos] + seq[pos + 1:]
    delta = f"del_{pos}_{old_aa}"
    return new_seq, "del", pos, delta

def mutate(
    seq: str,
    rng: np.random.Generator,
    p_sub: float = 0.70,
    p_ins: float = 0.15,
    p_del: float = 0.15,
) -> Tuple[str, str, int, str]:
    """
    Selects and executes a safe mutation operator based on length boundary conditions.
    Guarantees child length is strictly between [8, 50] residues.
    """
    seq = _validated_parent(seq)
    probabilities = np.asarray([p_sub, p_ins, p_del], dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Mutation probabilities must be finite and non-negative")
    operators = ["sub", "ins", "del"]
    allowed = [True, len(seq) < MAX_LEN, len(seq) > MIN_LEN]
    weights = probabilities * np.asarray(allowed, dtype=np.float64)
    if weights.sum() <= 0:
        # Preserve an available, length-safe mutation even if the caller assigned
        # all probability to an operator that the boundary makes impossible.
        weights[0] = 1.0
    weights /= weights.sum()
    op_choice = str(rng.choice(operators, p=weights))

    if op_choice == "sub":
        child_seq, op, pos, delta = point_substitution(seq, rng)
    elif op_choice == "ins":
        child_seq, op, pos, delta = insertion(seq, rng)
    else:
        child_seq, op, pos, delta = deletion(seq, rng)

    # Keep correctness checks active even when Python assertions are disabled.
    if not MIN_LEN <= len(child_seq) <= MAX_LEN:
        raise RuntimeError(f"Length invariant violated: {len(child_seq)}")
    if not set(child_seq) <= STANDARD_AA_SET:
        raise RuntimeError(f"Alphabet invariant violated: {child_seq}")

    return child_seq, op, pos, delta
