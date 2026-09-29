"""
Shared Evaluator Adapter and Call-Accounting Ledger (Role 05)
Tracks:
- Unique evaluator calls
- Cache hits
- Pre-evaluator rejections
- Multi-component biophysical & antimicrobial fitness
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import math

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "data-engineering/src"))
from amp_data.core import biophysical_descriptors
from amp_data.synthesis_filter import is_synthesizable

STANDARD_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

class EvaluatorAdapter:
    def __init__(self, version: str = "1.0.0", reference_sequences=None):
        self.version = version
        self.reference_sequences = {
            str(sequence).strip().upper() for sequence in (reference_sequences or set())
        }
        self.unique_evaluator_calls = 0
        self.cache_hits = 0
        self.pre_evaluator_rejections = 0
        self.total_requests = 0
        self.cache: Dict[str, Dict[str, Any]] = {}

    def reset_ledger(self):
        self.unique_evaluator_calls = 0
        self.cache_hits = 0
        self.pre_evaluator_rejections = 0
        self.total_requests = 0
        self.cache.clear()

    def evaluate(self, seq: str) -> Dict[str, Any]:
        """
        Evaluate using a hand-coded evolutionary fitness heuristic, not the Role-06 model.
        Checks cache before executing compute to strictly account for evaluator calls.
        """
        self.total_requests += 1
        seq_clean = seq.strip().upper()

        # 1. Pre-evaluator validity gate
        if not (8 <= len(seq_clean) <= 50) or not set(seq_clean).issubset(STANDARD_AA_SET):
            self.pre_evaluator_rejections += 1
            return {
                "sequence": seq_clean,
                "is_valid": False,
                "composite_fitness": -999.0,
                "amp_score": 0.0,
                "toxicity_risk": 1.0,
                "synthesizable": False,
                "synthesis_reason": "invalid_alphabet_or_length",
                "cache_hit": False,
            }

        if seq_clean in self.reference_sequences:
            self.pre_evaluator_rejections += 1
            desc = biophysical_descriptors(seq_clean)
            synth_ok, synth_reason = is_synthesizable(seq_clean)
            return {
                "sequence": seq_clean,
                "is_valid": False,
                "composite_fitness": -999.0,
                "amp_score": 0.0,
                "toxicity_risk": 1.0,
                "net_charge_ph7": round(float(desc.get("net_charge_ph7", 0.0)), 3),
                "eisenberg_moment": round(float(desc.get("eisenberg_hydrophobic_moment", 0.0)), 3),
                "boman_index": round(float(desc.get("boman_index", 0.0)), 3),
                "gravy": round(float(desc.get("grand_avg_hydropathy", 0.0)), 3),
                "instability_index": round(float(desc.get("instability_index", 0.0)), 3),
                "synthesizable": synth_ok,
                "synthesis_reason": "exact_official_reference_match; " + synth_reason,
                "cache_hit": False,
            }

        # 2. Check cache
        if seq_clean in self.cache:
            self.cache_hits += 1
            result = dict(self.cache[seq_clean])
            result["cache_hit"] = True
            return result

        # 3. New unique evaluation -> increments ledger
        self.unique_evaluator_calls += 1

        # Biophysical calculations
        desc = biophysical_descriptors(seq_clean)
        chg = desc.get("net_charge_ph7", 0.0)
        mom = desc.get("eisenberg_hydrophobic_moment", 0.0)
        bom = desc.get("boman_index", 0.0)
        grv = desc.get("grand_avg_hydropathy", 0.0)
        inst = desc.get("instability_index", 30.0)
        L = len(seq_clean)

        # Synthesizability screen
        synth_ok, synth_reason = is_synthesizable(seq_clean)

        # Multi-component fitness function
        # A. Charge component: target +2 to +7 (optimal +4 to +6)
        if chg < 1.0:
            charge_score = max(0.0, chg / 1.0) * 0.2
        elif 1.0 <= chg <= 6.0:
            charge_score = 0.2 + 0.8 * (chg - 1.0) / 5.0
        else: # chg > 6.0
            charge_score = max(0.5, 1.0 - (chg - 6.0) * 0.1)

        # B. Amphipathic moment component: target >= 0.40
        moment_score = 1.0 / (1.0 + math.exp(-10.0 * (mom - 0.45)))

        # C. Boman index component: penalize > 2.5 kcal/mol
        boman_penalty = 1.0 / (1.0 + math.exp(-3.0 * (bom - 2.5)))

        # D. Hydropathy component: penalize extreme hydrophobicity (GRAVY > 0.3)
        gravy_penalty = max(0.0, (grv - 0.3) * 1.5) if grv > 0.3 else max(0.0, (-0.8 - grv) * 0.8)

        # E. Toxicity/hemolysis risk estimate
        toxicity_risk = 0.5 * max(0.0, mom - 0.65) * 2.0 + 0.5 * max(0.0, grv - 0.2) * 2.0
        toxicity_risk = min(1.0, max(0.0, toxicity_risk))

        # Composite AMP likelihood score in [0, 1]
        amp_score = 0.45 * charge_score + 0.45 * moment_score - 0.10 * boman_penalty - 0.10 * gravy_penalty
        amp_score = min(1.0, max(0.0, amp_score))

        # Composite Fitness: balance activity vs toxicity vs synthesizability
        synth_penalty = 0.25 if not synth_ok else 0.0
        composite_fitness = amp_score - 0.35 * toxicity_risk - synth_penalty

        result = {
            "sequence": seq_clean,
            "is_valid": True,
            "composite_fitness": round(composite_fitness, 4),
            "amp_score": round(amp_score, 4),
            "toxicity_risk": round(toxicity_risk, 4),
            "net_charge_ph7": round(chg, 3),
            "eisenberg_moment": round(mom, 3),
            "boman_index": round(bom, 3),
            "gravy": round(grv, 3),
            "instability_index": round(inst, 3),
            "synthesizable": synth_ok,
            "synthesis_reason": synth_reason,
            "cache_hit": False,
        }

        self.cache[seq_clean] = result
        return result

    def get_ledger_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "unique_evaluator_calls": self.unique_evaluator_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / max(1, self.total_requests), 4),
            "pre_evaluator_rejections": self.pre_evaluator_rejections,
            "cached_sequences_count": len(self.cache),
            "evaluator_version": self.version,
        }
