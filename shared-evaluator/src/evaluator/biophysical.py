"""
Soft Biophysical Properties Profiler (Role 06)
Computes:
- Net Charge at pH 7.4 (Lehninger scale)
- Isoelectric Point (pI)
- Molecular Weight (Da)
- Eisenberg Hydrophobic Moment (alpha-helix delta = 100 deg)
- Kyte-Doolittle Grand Average of Hydropathy (GRAVY)
- Boman Protein-Binding Potential Index (kcal/mol)
- Instability Index
- Amino Acid Composition Breakdown
"""

import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "data-engineering/src"))
from amp_data.core import biophysical_descriptors

def compute_biophysical_properties(seq: str) -> Dict[str, Any]:
    """Calculate comprehensive biophysical profile for a canonical peptide sequence."""
    clean_seq = str(seq).strip().upper()
    L = len(clean_seq)
    
    # Base descriptors from vetted data-engineering core
    desc = biophysical_descriptors(clean_seq)
    
    # Composition breakdown
    cationic_count = clean_seq.count("K") + clean_seq.count("R") + clean_seq.count("H")
    anionic_count = clean_seq.count("D") + clean_seq.count("E")
    hydrophobic_count = sum(clean_seq.count(aa) for aa in "VILMFW")
    aromatic_count = clean_seq.count("F") + clean_seq.count("W") + clean_seq.count("Y")

    return {
        "length": L,
        "molecular_weight_da": desc.get("molecular_weight_da", 0.0),
        "net_charge_ph7": desc.get("net_charge_ph7", 0.0),
        "isoelectric_point": desc.get("isoelectric_point", 7.0),
        "eisenberg_moment": desc.get("eisenberg_hydrophobic_moment", 0.0),
        "gravy": desc.get("grand_avg_hydropathy", 0.0),
        "boman_index": desc.get("boman_index", 0.0),
        "instability_index": desc.get("instability_index", 0.0),
        "fraction_cationic": round(cationic_count / max(1, L), 4),
        "fraction_anionic": round(anionic_count / max(1, L), 4),
        "fraction_hydrophobic": round(hydrophobic_count / max(1, L), 4),
        "fraction_aromatic": round(aromatic_count / max(1, L), 4),
    }
