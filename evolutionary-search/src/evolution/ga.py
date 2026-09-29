"""
Genetic Algorithm (GA) Baseline for Evolutionary AMP Design (Role 05)
Features:
- Multi-family subpopulation representation
- Strict per-family quotas to guarantee structural & sequence diversity
- Tournament selection and boundary-safe mutation operators
- Complete parent-child lineage logging
- Budget-capped termination based strictly on unique evaluator calls
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .mutation import mutate
from .evaluator_adapter import EvaluatorAdapter


def validate_ancestry(candidates: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    """Raise when lineage IDs, parent links, or generation order are inconsistent."""
    nodes = {node["sequence_id"]: node for node in candidates}
    if len(nodes) != len(candidates):
        raise ValueError("Candidate sequence_id values must be unique")
    seed_nodes = {
        node["sequence_id"] for node in candidates
        if node.get("parent_id") is None and int(node["generation"]) == 0
    }
    for node in candidates:
        if node.get("parent_id") is None and int(node["generation"]) != 0:
            raise ValueError(f"Root candidate must have generation 0: {node['sequence_id']}")
        if node.get("seed_id") not in seed_nodes:
            raise ValueError(f"Candidate references an unknown seed root: {node['sequence_id']}")
    edge_children = set()
    for edge in edges:
        parent_id, child_id = edge.get("parent_id"), edge.get("child_id")
        if parent_id not in nodes or child_id not in nodes:
            raise ValueError(f"Ancestry edge references an unknown node: {edge}")
        if child_id in edge_children:
            raise ValueError(f"Candidate has more than one parent edge: {child_id}")
        parent, child = nodes[parent_id], nodes[child_id]
        if child.get("parent_id") != parent_id:
            raise ValueError(f"Candidate parent field disagrees with its edge: {child_id}")
        if int(edge.get("generation", -1)) != int(child["generation"]):
            raise ValueError(f"Ancestry edge generation disagrees with its child: {edge}")
        if int(parent["generation"]) >= int(child["generation"]):
            raise ValueError(f"Ancestry generation must increase from parent to child: {edge}")
        if parent.get("seed_id") != child.get("seed_id") or edge.get("seed_id") != child.get("seed_id"):
            raise ValueError(f"Ancestry edge crosses seed families: {edge}")
        edge_children.add(child_id)

    expected_children = {
        node["sequence_id"] for node in candidates if node.get("parent_id") is not None
    }
    if edge_children != expected_children:
        raise ValueError(
            f"Ancestry DAG has missing/extra child edges: "
            f"missing={len(expected_children - edge_children)}, extra={len(edge_children - expected_children)}"
        )

class GeneticAlgorithmSearch:
    def __init__(
        self,
        seeds: List[Dict[str, Any]],
        evaluator: EvaluatorAdapter,
        budget: int = 1000,
        pop_size_per_family: int = 5,
        offspring_per_parent: int = 3,
        tournament_k: int = 3,
        run_id: str = "ga_run_001",
        seed: int = 42,
        max_generations: int = 500,
        max_stagnant_generations: int = 10,
    ):
        if budget <= 0 or pop_size_per_family <= 0 or offspring_per_parent <= 0:
            raise ValueError("budget, pop_size_per_family, and offspring_per_parent must be positive")
        if tournament_k <= 0 or max_generations <= 0 or max_stagnant_generations <= 0:
            raise ValueError("tournament_k and generation limits must be positive")
        if not seeds:
            raise ValueError("At least one seed sequence is required")
        if len(seeds) > budget:
            raise ValueError(f"Seed count {len(seeds)} exceeds unique-evaluation budget {budget}")
        seed_sequences = [str(seed.get("sequence", "")).strip().upper() for seed in seeds]
        if len(set(seed_sequences)) != len(seed_sequences):
            raise ValueError("Seed sequences must be unique")
        for sequence in seed_sequences:
            if not 8 <= len(sequence) <= 50 or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
                raise ValueError(f"Invalid seed peptide: {sequence!r}")

        self.seeds = [dict(seed, sequence=sequence) for seed, sequence in zip(seeds, seed_sequences)]
        self.evaluator = evaluator
        self.budget = budget
        self.pop_size_per_family = pop_size_per_family
        self.offspring_per_parent = offspring_per_parent
        self.tournament_k = tournament_k
        self.run_id = run_id
        self.rng = np.random.default_rng(seed)
        self.max_generations = max_generations
        self.max_stagnant_generations = max_stagnant_generations

        self.candidates_pool: List[Dict[str, Any]] = []
        self.ancestry_edges: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.cand_counter = 0

    def run(self) -> Dict[str, Any]:
        """Execute the GA search until the unique evaluator call budget is exhausted."""
        # 1. Initialize Generation 0 with seeds
        current_pop: List[Dict[str, Any]] = []
        for s in self.seeds:
            seq = s["sequence"]
            ev = self.evaluator.evaluate(seq)
            s_id = f"ga_seed_{self.cand_counter:05d}"
            self.cand_counter += 1

            node = {
                "sequence_id": s_id,
                "sequence": seq,
                "domain": "evolution",
                "model": "GA",
                "run_id": self.run_id,
                "generation": 0,
                "parent_id": None,
                "seed_id": s_id,
                "seed_family": int(s.get("family_id", 0)),
                "family_archetype": s.get("family_archetype", "Unknown"),
                "mutation_type": "seed",
                "mutation_pos": -1,
                "mutation_desc": "initial_seed",
                "is_valid": bool(ev.get("is_valid", True)),
                "composite_fitness": ev["composite_fitness"],
                "amp_score": ev["amp_score"],
                "toxicity_risk": ev["toxicity_risk"],
                "net_charge_ph7": ev["net_charge_ph7"],
                "eisenberg_moment": ev["eisenberg_moment"],
                "boman_index": ev["boman_index"],
                "gravy": ev["gravy"],
                "instability_index": ev["instability_index"],
                "synthesizable": ev["synthesizable"],
                "synthesis_reason": ev["synthesis_reason"],
            }
            current_pop.append(node)
            self.candidates_pool.append(node)

        gen = 0
        stagnant_generations = 0
        termination_reason = "budget_exhausted"
        while (
            self.evaluator.unique_evaluator_calls < self.budget
            and gen < self.max_generations
        ):
            gen += 1
            # Check if budget exhausted
            if self.evaluator.unique_evaluator_calls >= self.budget:
                break

            calls_before_generation = self.evaluator.unique_evaluator_calls
            gen_proposals: List[Dict[str, Any]] = []

            # Group current population by family
            families = {}
            for ind in current_pop:
                families.setdefault(ind["seed_family"], []).append(ind)

            # Produce offspring for each family
            for fam_id, members in families.items():
                if self.evaluator.unique_evaluator_calls >= self.budget:
                    break

                for _ in range(len(members) * self.offspring_per_parent):
                    if self.evaluator.unique_evaluator_calls >= self.budget:
                        break

                    # Tournament selection
                    k = min(self.tournament_k, len(members))
                    idx_choices = self.rng.choice(len(members), size=k, replace=False)
                    best_parent = max((members[i] for i in idx_choices), key=lambda x: x["composite_fitness"])

                    # Mutate
                    child_seq, op, pos, delta = mutate(best_parent["sequence"], self.rng)
                    ev = self.evaluator.evaluate(child_seq)
                    if ev.get("cache_hit", False) or not ev.get("is_valid", True):
                        continue

                    c_id = f"ga_cand_{self.cand_counter:05d}"
                    self.cand_counter += 1

                    child_node = {
                        "sequence_id": c_id,
                        "sequence": child_seq,
                        "domain": "evolution",
                        "model": "GA",
                        "run_id": self.run_id,
                        "generation": gen,
                        "parent_id": best_parent["sequence_id"],
                        "seed_id": best_parent["seed_id"],
                        "seed_family": fam_id,
                        "family_archetype": best_parent["family_archetype"],
                        "mutation_type": op,
                        "mutation_pos": pos,
                        "mutation_desc": delta,
                        "is_valid": bool(ev.get("is_valid", True)),
                        "composite_fitness": ev["composite_fitness"],
                        "amp_score": ev["amp_score"],
                        "toxicity_risk": ev["toxicity_risk"],
                        "net_charge_ph7": ev["net_charge_ph7"],
                        "eisenberg_moment": ev["eisenberg_moment"],
                        "boman_index": ev["boman_index"],
                        "gravy": ev["gravy"],
                        "instability_index": ev["instability_index"],
                        "synthesizable": ev["synthesizable"],
                        "synthesis_reason": ev["synthesis_reason"],
                    }
                    gen_proposals.append(child_node)
                    self.candidates_pool.append(child_node)

                    # Log ancestry edge
                    self.ancestry_edges.append({
                        "parent_id": best_parent["sequence_id"],
                        "child_id": c_id,
                        "seed_id": best_parent["seed_id"],
                        "generation": gen,
                        "mutation_type": op,
                        "position": pos,
                        "delta": delta,
                        "fitness_delta": round(ev["composite_fitness"] - best_parent["composite_fitness"], 4),
                    })

            # Selection with strict family quotas (elitism + top offspring per family)
            if gen_proposals:
                survivors = []
                combined_pool = current_pop + gen_proposals
                df_combined = pd.DataFrame(combined_pool)

                for fam_id, group in df_combined.groupby("seed_family"):
                    # Take top pop_size_per_family per family
                    top_fam = group.sort_values(by="composite_fitness", ascending=False).head(self.pop_size_per_family)
                    survivors.extend(top_fam.to_dict(orient="records"))

                current_pop = survivors

            # Log generation metrics
            fits = [ind["composite_fitness"] for ind in current_pop]
            fam_counts = pd.Series([ind["seed_family"] for ind in current_pop]).value_counts().to_dict()
            self.history.append({
                "generation": gen,
                "mean_fitness": round(float(np.mean(fits)), 4),
                "max_fitness": round(float(np.max(fits)), 4),
                "active_families": len(fam_counts),
                "evaluator_calls": self.evaluator.unique_evaluator_calls,
                "pool_size": len(self.candidates_pool),
            })

            if self.evaluator.unique_evaluator_calls == calls_before_generation:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
            if stagnant_generations >= self.max_stagnant_generations:
                termination_reason = "stagnation_limit"
                break

        if self.evaluator.unique_evaluator_calls >= self.budget:
            termination_reason = "budget_exhausted"
        elif gen >= self.max_generations and termination_reason != "stagnation_limit":
            termination_reason = "max_generations"

        validate_ancestry(self.candidates_pool, self.ancestry_edges)
        return {
            "algorithm": "GeneticAlgorithm",
            "generations": gen,
            "candidates": self.candidates_pool,
            "ancestry_edges": self.ancestry_edges,
            "history": self.history,
            "ledger": self.evaluator.get_ledger_stats(),
            "termination_reason": termination_reason,
        }
