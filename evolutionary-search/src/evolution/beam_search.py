"""
Beam Search Challenger for Evolutionary AMP Design (Role 05)
Features:
- Bounded beam expansion with matched mutation operators
- Family-diversity quotas on beam survivors
- Identical evaluator ledger accounting
- Complete parent-child lineage logging
- Budget-capped termination based strictly on unique evaluator calls
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .mutation import mutate
from .evaluator_adapter import EvaluatorAdapter

class BeamSearchChallenger:
    def __init__(
        self,
        seeds: List[Dict[str, Any]],
        evaluator: EvaluatorAdapter,
        budget: int = 1000,
        beam_width: int = 40,
        expansions_per_parent: int = 5,
        beam_quota_per_family: int = 5,
        run_id: str = "beam_run_001",
        seed: int = 42,
    ):
        self.seeds = seeds
        self.evaluator = evaluator
        self.budget = budget
        self.beam_width = beam_width
        self.expansions_per_parent = expansions_per_parent
        self.beam_quota_per_family = beam_quota_per_family
        self.run_id = run_id
        self.rng = np.random.default_rng(seed)

        self.candidates_pool: List[Dict[str, Any]] = []
        self.ancestry_edges: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.cand_counter = 0

    def run(self) -> Dict[str, Any]:
        """Execute the Beam Search until the unique evaluator call budget is exhausted."""
        # 1. Initialize Beam with seeds
        current_beam: List[Dict[str, Any]] = []
        for s in self.seeds:
            seq = s["sequence"]
            ev = self.evaluator.evaluate(seq)
            s_id = f"beam_seed_{self.cand_counter:05d}"
            self.cand_counter += 1

            node = {
                "sequence_id": s_id,
                "sequence": seq,
                "domain": "evolution",
                "model": "BeamSearch",
                "run_id": self.run_id,
                "round": 0,
                "parent_id": None,
                "seed_id": s_id,
                "seed_family": int(s.get("family_id", 0)),
                "family_archetype": s.get("family_archetype", "Unknown"),
                "mutation_type": "seed",
                "mutation_pos": -1,
                "mutation_desc": "initial_seed",
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
            current_beam.append(node)
            self.candidates_pool.append(node)

        round_idx = 0
        while self.evaluator.unique_evaluator_calls < self.budget:
            round_idx += 1
            if self.evaluator.unique_evaluator_calls >= self.budget:
                break

            proposals: List[Dict[str, Any]] = []

            # Expand each parent in the beam
            for parent in current_beam:
                if self.evaluator.unique_evaluator_calls >= self.budget:
                    break

                for _ in range(self.expansions_per_parent):
                    if self.evaluator.unique_evaluator_calls >= self.budget:
                        break

                    child_seq, op, pos, delta = mutate(parent["sequence"], self.rng)
                    ev = self.evaluator.evaluate(child_seq)

                    c_id = f"beam_cand_{self.cand_counter:05d}"
                    self.cand_counter += 1

                    child_node = {
                        "sequence_id": c_id,
                        "sequence": child_seq,
                        "domain": "evolution",
                        "model": "BeamSearch",
                        "run_id": self.run_id,
                        "round": round_idx,
                        "parent_id": parent["sequence_id"],
                        "seed_id": parent["seed_id"],
                        "seed_family": parent["seed_family"],
                        "family_archetype": parent["family_archetype"],
                        "mutation_type": op,
                        "mutation_pos": pos,
                        "mutation_desc": delta,
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
                    proposals.append(child_node)
                    self.candidates_pool.append(child_node)

                    # Log ancestry edge
                    self.ancestry_edges.append({
                        "parent_id": parent["sequence_id"],
                        "child_id": c_id,
                        "seed_id": parent["seed_id"],
                        "round": round_idx,
                        "mutation_type": op,
                        "position": pos,
                        "delta": delta,
                        "fitness_delta": round(ev["composite_fitness"] - parent["composite_fitness"], 4),
                    })

            if not proposals:
                break

            # Beam Selection with family quotas
            all_cands = current_beam + proposals
            df_all = pd.DataFrame(all_cands)

            survivors = []
            for fam_id, group in df_all.groupby("seed_family"):
                top_fam = group.sort_values(by="composite_fitness", ascending=False).head(self.beam_quota_per_family)
                survivors.extend(top_fam.to_dict(orient="records"))

            # Sort all survivors and take top beam_width
            df_surv = pd.DataFrame(survivors)
            df_top = df_surv.sort_values(by="composite_fitness", ascending=False).head(self.beam_width)
            current_beam = df_top.to_dict(orient="records")

            # Log round metrics
            fits = [b["composite_fitness"] for b in current_beam]
            fam_counts = pd.Series([b["seed_family"] for b in current_beam]).value_counts().to_dict()
            self.history.append({
                "round": round_idx,
                "mean_fitness": round(float(np.mean(fits)), 4),
                "max_fitness": round(float(np.max(fits)), 4),
                "active_families": len(fam_counts),
                "evaluator_calls": self.evaluator.unique_evaluator_calls,
                "pool_size": len(self.candidates_pool),
            })

        return {
            "algorithm": "BeamSearch",
            "rounds": round_idx,
            "candidates": self.candidates_pool,
            "ancestry_edges": self.ancestry_edges,
            "history": self.history,
            "ledger": self.evaluator.get_ledger_stats(),
        }
