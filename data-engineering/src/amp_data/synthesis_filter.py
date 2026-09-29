"""Lightweight sequence pre-filter for generator loops.

Hard failures return ``False``. A lone cysteine is a soft warning and returns
``(True, 'warn_unpaired_cys')`` so callers can retain or reject it by policy.
"""
from __future__ import annotations

import re
import pandas as pd
from .core import ALPHABET, eisenberg_hydrophobic_moment, net_charge_ph7

HYDROPHOBIC = "VILFWM"

def is_synthesizable(sequence: str) -> tuple[bool, str]:
    sequence = str(sequence or "").strip().upper()
    if not sequence or set(sequence) - ALPHABET:
        return False, "invalid_residue"
    if not 8 <= len(sequence) <= 50:
        return False, "length_out_of_range"
    if re.search(f"[{HYDROPHOBIC}]{{5,}}", sequence):
        return False, "hydrophobic_run"
    if re.search(r"(.)\1{3,}", sequence):
        return False, "polyrepeat"
    charge = net_charge_ph7(sequence)
    if charge is None:
        return False, "insufficient_charge"
    if eisenberg_hydrophobic_moment(sequence) > .6 and charge < 1:
        return False, "aggregation_risk"
    if charge is None or charge < 1:
        return False, "insufficient_charge"
    if sequence.count("C") == 1:
        return True, "warn_unpaired_cys"
    return True, "ok"

def filter_sequences(sequences: list[str]) -> pd.DataFrame:
    """Apply :func:`is_synthesizable` to a batch for generator-side filtering."""
    records = []
    for sequence in sequences:
        ok, reason = is_synthesizable(sequence)
        records.append({"sequence": sequence, "is_valid": ok, "rejection_reason": reason})
    return pd.DataFrame(records)
