"""
Unified Candidate Validator and Hard Rules Gate (Role 06)
Implements:
- Strict alphabet validation (20 standard proteinogenic amino acids)
- Strict length validation (8 <= L <= 50)
- Exact reference match quarantine (against antibacterial.fasta)
- Schema structure validation
"""

from typing import Tuple, List, Set, Dict, Any

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_SET = set(STANDARD_AA)
MIN_LEN = 8
MAX_LEN = 50

REQUIRED_FIELDS = {"sequence_id", "sequence", "domain", "model"}

def validate_schema(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that candidate dictionary contains all mandatory schema fields."""
    errors = []
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        errors.append(f"missing_required_fields: {sorted(list(missing))}")
    if not isinstance(record.get("sequence"), str) or len(record.get("sequence", "")) == 0:
        errors.append("empty_or_non_string_sequence")
    return len(errors) == 0, errors

def validate_hard_rules(seq: str, reference_set: Set[str] = None) -> Tuple[bool, List[str]]:
    """
    Enforce official challenge hard constraints:
    1. 20 standard canonical proteinogenic amino acids only
    2. Length strictly within [8, 50] residues
    3. Exactly 0 exact matches to reference sequences
    """
    errors = []
    clean_seq = str(seq).strip().upper()
    L = len(clean_seq)

    # 1. Alphabet Check
    invalid_chars = set(clean_seq) - STANDARD_AA_SET
    if invalid_chars:
        errors.append(f"non_canonical_amino_acids: {sorted(list(invalid_chars))}")

    # 2. Length Check
    if L < MIN_LEN:
        errors.append(f"length_below_minimum: length={L} < {MIN_LEN}")
    elif L > MAX_LEN:
        errors.append(f"length_above_maximum: length={L} > {MAX_LEN}")

    # 3. Exact Reference Overlap Check
    if reference_set and clean_seq in reference_set:
        errors.append("exact_match_to_official_reference")

    return len(errors) == 0, errors
