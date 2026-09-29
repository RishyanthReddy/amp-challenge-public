"""
Unit Tests for Evolutionary Mutation Operators (Role 05)
Verifies:
- Length invariant [8, 50] under boundary conditions
- 100% canonical amino acids
- Correct operation tracking and delta reporting
"""

import sys
from pathlib import Path
import numpy as np
import pytest
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(SRC_DIR))
from evolution.mutation import (
    point_substitution,
    insertion,
    deletion,
    mutate,
    STANDARD_AA_SET,
    MIN_LEN,
    MAX_LEN,
)

def test_point_substitution():
    rng = np.random.default_rng(42)
    seq = "KWKLFKKIGKVLKVL"
    child, op, pos, delta = point_substitution(seq, rng)
    assert len(child) == len(seq)
    assert op == "sub"
    assert 0 <= pos < len(seq)
    assert child[pos] != seq[pos]
    assert set(child).issubset(STANDARD_AA_SET)
    assert delta == f"{seq[pos]}{pos}{child[pos]}"

def test_deletion_boundary_min_len():
    rng = np.random.default_rng(123)
    seq_8 = "KIWVIRWR"  # exactly 8 residues
    assert len(seq_8) == 8

    # When deletion is explicitly requested on len 8, it must guard and not shrink below 8
    child, op, pos, delta = deletion(seq_8, rng)
    assert len(child) >= 8

    # Using mutate() wrapper at len 8
    for _ in range(100):
        child, op, pos, delta = mutate(seq_8, rng)
        assert len(child) >= 8
        assert set(child).issubset(STANDARD_AA_SET)

def test_insertion_boundary_max_len():
    rng = np.random.default_rng(456)
    seq_50 = "A" * 50  # exactly 50 residues
    assert len(seq_50) == 50

    # When insertion is explicitly requested on len 50, it must guard and not exceed 50
    child, op, pos, delta = insertion(seq_50, rng)
    assert len(child) <= 50

    # Using mutate() wrapper at len 50
    for _ in range(100):
        child, op, pos, delta = mutate(seq_50, rng)
        assert len(child) <= 50
        assert set(child).issubset(STANDARD_AA_SET)

def test_stress_random_mutations():
    rng = np.random.default_rng(999)
    seeds = [
        "KIWVIRWR",            # L=8
        "LIKHILHRL",           # L=9
        "KWKLFKKIGKVLKVL",     # L=15
        "AALKGCWTKSIPPKPCFGKR", # L=20
        "KYYGNGVTCGKHSCSVDWGKATTCIINNGAAAWATGGHQGNHKC", # L=44
        "M" * 50,              # L=50
    ]

    for seed in seeds:
        cur = seed
        for _ in range(500):
            cur, op, pos, delta = mutate(cur, rng)
            assert 8 <= len(cur) <= 50
            assert set(cur).issubset(STANDARD_AA_SET)
            assert op in {"sub", "ins", "del"}
            assert isinstance(pos, int)
            assert isinstance(delta, str)

def test_mutation_rejects_invalid_parents_and_probabilities():
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="parent length"):
        mutate("AAAAAAA", rng)
    with pytest.raises(ValueError, match="noncanonical"):
        mutate("AAAAAAAX", rng)
    with pytest.raises(ValueError, match="probabilities"):
        mutate("A" * 8, rng, p_sub=-1, p_ins=1, p_del=1)

def test_boundary_probabilities_fall_back_to_safe_operator():
    rng = np.random.default_rng(13)
    child, op, _, _ = mutate("A" * 50, rng, p_sub=0, p_ins=1, p_del=0)
    assert len(child) == 50
    assert op == "sub"


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])
