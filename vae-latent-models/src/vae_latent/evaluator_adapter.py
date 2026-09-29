"""
Role 03: VAE & Latent Models — Shared Evaluator Integration Adapter
Bridges HydrAMP generated candidate pools to the team's shared evaluation standards.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "data-engineering/src"))
from amp_data.core import biophysical_descriptors
from amp_data.synthesis_filter import is_synthesizable

def evaluate_vae_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies shared biophysical descriptors, biological synthesizability screens,
    and validity flags to any VAE candidate dataframe.
    """
    STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")
    evaluated_records = []

    for _, row in candidates_df.iterrows():
        seq = str(row["sequence"]).strip().upper()
        L = len(seq)
        is_canonical = set(seq).issubset(STANDARD_AAS)
        len_ok = 8 <= L <= 50
        is_valid = is_canonical and len_ok and (L > 0)

        synth_ok, synth_reason = is_synthesizable(seq)
        desc = biophysical_descriptors(seq) if is_valid else {}

        record = dict(row)
        record.update({
            "sequence": seq,
            "length": L,
            "is_valid": is_valid,
            "valid_standard_aa": is_canonical,
            "synthesizable": synth_ok,
            "synthesis_flag": synth_reason,
            "net_charge_ph7": desc.get("net_charge_ph7", None),
            "eisenberg_moment": desc.get("eisenberg_hydrophobic_moment", None),
            "boman_index": desc.get("boman_index", None),
            "gravy": desc.get("grand_avg_hydropathy", None),
            "instability_index": desc.get("instability_index", None),
        })
        evaluated_records.append(record)

    return pd.DataFrame(evaluated_records)
