"""
MAP-Elites Quality-Diversity Archive (Role 07)
Discretizes peptide space across 3 phenotypic/biophysical axes:
1. Net Charge at pH 7.4 (6 bins)
2. Eisenberg Hydrophobic Moment (5 bins)
3. Length (5 bins)
Total: 150 niche cells.
    Selects cell elites by a conservative composite score:
      q_i = P_AMP - 0.5 R_tox - sigma_AMP + MIC bonus + consensus bonus
           - synthesis penalty

    The score is an LCB-style heuristic. Its uncertainty term is only as calibrated as
    the upstream model's ensemble disagreement estimate.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

class MapElitesArchive:
    def __init__(self):
        # Charge bins: [ <1, 1-3, 3-5, 5-7, 7-10, >=10 ]
        self.charge_bins = [-np.inf, 1.0, 3.0, 5.0, 7.0, 10.0, np.inf]
        # Moment bins: [ <0.30, 0.30-0.45, 0.45-0.60, 0.60-0.75, >=0.75 ]
        self.moment_bins = [-np.inf, 0.30, 0.45, 0.60, 0.75, np.inf]
        # Length bins: [ 8-14, 15-22, 23-30, 31-40, 41-50 ]
        # np.digitize uses half-open bins with right=False, so these integer edges
        # put lengths 8-14, 15-22, 23-30, 31-40, and 41-50 in the documented bins.
        self.length_bins = [7, 15, 23, 31, 41, 51]

        self.n_charge = len(self.charge_bins) - 1
        self.n_moment = len(self.moment_bins) - 1
        self.n_length = len(self.length_bins) - 1
        self.total_cells = self.n_charge * self.n_moment * self.n_length  # 150

        self.grid: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

    def get_cell_coordinates(self, charge: float, moment: float, length: int) -> Tuple[int, int, int]:
        if not np.isfinite(charge) or not np.isfinite(moment):
            raise ValueError("MAP-Elites charge and moment must be finite")
        if not 8 <= int(length) <= 50:
            raise ValueError(f"MAP-Elites length must be in [8, 50], got {length}")
        c_idx = int(np.digitize([charge], self.charge_bins)[0]) - 1
        m_idx = int(np.digitize([moment], self.moment_bins)[0]) - 1
        l_idx = int(np.digitize([length], self.length_bins)[0]) - 1
        return c_idx, m_idx, l_idx

    @staticmethod
    def calculate_lcb_quality(row: Dict[str, Any]) -> float:
        """Conservative quality score penalizing epistemic uncertainty & toxicity."""
        required = ("pred_amp_probability", "pred_toxicity_risk", "uncertainty_amp_std", "synthesizable")
        missing = [key for key in required if key not in row or pd.isna(row[key])]
        if missing:
            raise ValueError(f"Cannot score candidate with missing quality fields: {missing}")
        p_amp = float(row["pred_amp_probability"])
        r_tox = float(row["pred_toxicity_risk"])
        sigma_amp = float(row["uncertainty_amp_std"])
        synth_ok = bool(row["synthesizable"])
        if not all(np.isfinite(x) for x in (p_amp, r_tox, sigma_amp)):
            raise ValueError("MAP-Elites quality inputs must be finite")
        if not 0.0 <= p_amp <= 1.0 or not 0.0 <= r_tox <= 1.0 or sigma_amp < 0.0:
            raise ValueError("MAP-Elites probabilities must be in [0, 1] and uncertainty non-negative")

        # APEX MIC bonus if available (lower MIC = higher potency)
        apex_mic = row.get("apex_mean_mic", None)
        mic_bonus = 0.0
        if apex_mic is not None and not np.isnan(float(apex_mic)):
            # Cap bonus at +0.2 for potent peptides <= 32 uM
            mic_val = float(apex_mic)
            if mic_val <= 32.0:
                mic_bonus = 0.20 * (1.0 - mic_val / 32.0)

        # Consensus multi-domain bonus
        domain_count = int(row.get("domain_count", 1))
        consensus_bonus = 0.05 if domain_count > 1 else 0.0

        synth_penalty = 0.20 if not synth_ok else 0.0
        quality = (p_amp - 0.5 * r_tox - 1.0 * sigma_amp + mic_bonus + consensus_bonus - synth_penalty)
        return float(np.round(quality, 4))

    def build_archive(self, candidates_df: pd.DataFrame) -> Dict[str, Any]:
        """Ingest candidate pool and place each sequence into its MAP-Elites cell."""
        self.grid.clear()
        records_with_cells = []

        for idx, row in candidates_df.iterrows():
            chg = float(row.get("net_charge_ph7", 0.0))
            mom = float(row.get("eisenberg_moment", 0.0))
            length = int(row.get("length", len(row["sequence"])))
            
            coord = self.get_cell_coordinates(chg, mom, length)
            quality = self.calculate_lcb_quality(row.to_dict())

            cand_entry = dict(row)
            cand_entry["map_cell"] = f"c{coord[0]}_m{coord[1]}_l{coord[2]}"
            cand_entry["cell_coords"] = coord
            cand_entry["lcb_quality"] = quality

            self.grid.setdefault(coord, []).append(cand_entry)
            records_with_cells.append(cand_entry)

        # Sort each cell by quality descending
        for coord in self.grid:
            self.grid[coord].sort(key=lambda x: x["lcb_quality"], reverse=True)

        occupied_cells = len(self.grid)
        coverage_pct = (occupied_cells / self.total_cells) * 100.0

        # Extract cell elites
        cell_elites = [self.grid[c][0] for c in self.grid]

        return {
            "occupied_cells": occupied_cells,
            "total_cells": self.total_cells,
            "coverage_percentage": round(coverage_pct, 2),
            "cell_elites": cell_elites,
            "annotated_df": pd.DataFrame(records_with_cells),
        }
