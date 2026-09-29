from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import evolution.ga as ga_module
from evolution.ga import GeneticAlgorithmSearch, validate_ancestry


class TinyEvaluator:
    def __init__(self):
        self.unique_evaluator_calls = 0
        self.cache = {}

    def evaluate(self, sequence):
        if sequence in self.cache:
            return {**self.cache[sequence], "cache_hit": True}
        self.unique_evaluator_calls += 1
        result = {
            "is_valid": True,
            "composite_fitness": len(set(sequence)) / len(sequence),
            "amp_score": 0.5,
            "toxicity_risk": 0.2,
            "net_charge_ph7": 2.0,
            "eisenberg_moment": 0.4,
            "boman_index": 1.0,
            "gravy": 0.0,
            "instability_index": 20.0,
            "synthesizable": True,
            "synthesis_reason": "pass",
            "cache_hit": False,
        }
        self.cache[sequence] = result
        return result

    def get_ledger_stats(self):
        return {"unique_evaluator_calls": self.unique_evaluator_calls}


def test_ga_respects_budget_and_returns_valid_lineage():
    evaluator = TinyEvaluator()
    search = GeneticAlgorithmSearch(
        seeds=[{"sequence": "ACDEFGHI", "family_id": 0}, {"sequence": "KLMNPQRS", "family_id": 1}],
        evaluator=evaluator,
        budget=8,
        pop_size_per_family=2,
        offspring_per_parent=2,
        tournament_k=2,
        seed=7,
    )
    result = search.run()
    assert result["ledger"]["unique_evaluator_calls"] <= 8
    assert result["termination_reason"] in {"budget_exhausted", "stagnation_limit", "max_generations"}
    validate_ancestry(result["candidates"], result["ancestry_edges"])


def test_ga_stagnation_guard_stops_repeated_cached_children(monkeypatch):
    monkeypatch.setattr(ga_module, "mutate", lambda seq, rng: (seq, "sub", 0, "same"))
    search = GeneticAlgorithmSearch(
        seeds=[{"sequence": "ACDEFGHI", "family_id": 0}],
        evaluator=TinyEvaluator(),
        budget=10,
        offspring_per_parent=1,
        tournament_k=1,
        max_stagnant_generations=2,
    )
    result = search.run()
    assert result["termination_reason"] == "stagnation_limit"
    assert result["generations"] == 2
    validate_ancestry(result["candidates"], result["ancestry_edges"])


def test_ancestry_validator_rejects_generation_back_edges():
    candidates = [
        {"sequence_id": "a", "generation": 1, "parent_id": "b", "seed_id": "c"},
        {"sequence_id": "b", "generation": 2, "parent_id": "c", "seed_id": "c"},
        {"sequence_id": "c", "generation": 0, "parent_id": None, "seed_id": "c"},
    ]
    edges = [
        {"parent_id": "c", "child_id": "b", "seed_id": "c", "generation": 2},
        {"parent_id": "b", "child_id": "a", "seed_id": "c", "generation": 1},
    ]
    with pytest.raises(ValueError, match="generation must increase"):
        validate_ancestry(candidates, edges)
