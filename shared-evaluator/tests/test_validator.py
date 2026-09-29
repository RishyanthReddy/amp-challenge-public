"""
Unit Tests for Evaluator Hard Rules & Schema Validator (Role 06)
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
from evaluator.validator import validate_schema, validate_hard_rules

def test_valid_peptide():
    seq = "KWKLFKKIGKVLKVL"
    ok, errors = validate_hard_rules(seq)
    assert ok is True
    assert len(errors) == 0

def test_length_boundary_conditions():
    # 7 residues fails
    ok_7, err_7 = validate_hard_rules("AAAAAAA")
    assert ok_7 is False
    assert any("length_below_minimum" in e for e in err_7)

    # 8 residues passes
    ok_8, err_8 = validate_hard_rules("AAAAAAAA")
    assert ok_8 is True

    # 50 residues passes
    ok_50, err_50 = validate_hard_rules("A" * 50)
    assert ok_50 is True

    # 51 residues fails
    ok_51, err_51 = validate_hard_rules("A" * 51)
    assert ok_51 is False
    assert any("length_above_maximum" in e for e in err_51)

def test_non_canonical_amino_acids():
    bad_seqs = [
        "KWKLFKKIGKVLKVX",  # X
        "KWKLFKKIGKVLKVU",  # Selenocysteine U
        "KWKLFKKIGKVLKV1",  # Digit
        "KWKLFKK-GKVLKVL",  # Dash
        "KWKLFKK BKVLKVL",  # Space & B
    ]
    for b in bad_seqs:
        ok, errors = validate_hard_rules(b)
        assert ok is False
        assert any("non_canonical_amino_acids" in e for e in errors)

def test_reference_exact_match():
    ref_set = {"PEPTIDEREFERENCE"}
    ok, errors = validate_hard_rules("PEPTIDEREFERENCE", reference_set=ref_set)
    assert ok is False
    assert "exact_match_to_official_reference" in errors

    ok_clean, _ = validate_hard_rules("CLEANNEWPEPTIDE", reference_set=ref_set)
    assert ok_clean is True

def test_schema_validation():
    valid_rec = {
        "sequence_id": "test_001",
        "sequence": "KWKLFKKIGKVLKVL",
        "domain": "autoregressive",
        "model": "ESM2_GPT",
    }
    ok, errs = validate_schema(valid_rec)
    assert ok is True

    invalid_rec = {
        "sequence_id": "test_002",
        # missing sequence, domain, model
    }
    ok_inv, errs_inv = validate_schema(invalid_rec)
    assert ok_inv is False
    assert len(errs_inv) > 0

if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
