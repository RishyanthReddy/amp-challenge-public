"""
Unit Tests for Biophysical Properties Module (Role 06)
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
from evaluator.biophysical import compute_biophysical_properties

def test_biophysical_properties_calculation():
    seq = "KWKLFKKIGKVLKVL"  # 15-aa highly cationic amphipathic peptide
    props = compute_biophysical_properties(seq)

    assert props["length"] == 15
    assert props["molecular_weight_da"] > 1000.0
    assert props["net_charge_ph7"] > 4.0  # contains 5 Lysines
    assert props["eisenberg_moment"] > 0.50  # amphipathic
    assert 0.0 <= props["fraction_cationic"] <= 1.0
    assert 0.0 <= props["fraction_hydrophobic"] <= 1.0
    assert isinstance(props["gravy"], float)
    assert isinstance(props["boman_index"], float)

def test_charge_neutral_peptide():
    seq = "GAGAGAGAGAGAGAGA"  # 16-aa glycine-alanine repeating
    props = compute_biophysical_properties(seq)
    assert props["length"] == 16
    assert abs(props["net_charge_ph7"]) < 0.5  # neutral termini

if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
