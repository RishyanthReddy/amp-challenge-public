"""
Activity and Safety Predictors with Explicit Uncertainty Estimation (Role 06)
Implements:
1. Feature extraction from sequence (AAC, charge, moment, Boman, hydropathy)
2. Ensemble-based AMP likelihood predictor with epistemic uncertainty (sigma_amp)
3. Empirical Hemolysis probability predictor (trained on DBAASP RBC assays) with uncertainty (sigma_hemo)
4. Biophysical membrane disruption risk proxy
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "data-engineering/src"))
from amp_data.core import biophysical_descriptors

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except:
        return default

def extract_features(seq: str) -> np.ndarray:
    """Extract a 27-dimensional biophysical and compositional feature vector."""
    clean_seq = str(seq).strip().upper()
    L = max(1, len(clean_seq))
    
    # 20 amino acid frequencies
    aac = [clean_seq.count(aa) / L for aa in STANDARD_AA]
    
    # 7 biophysical scalar descriptors
    desc = biophysical_descriptors(clean_seq) if len(clean_seq) >= 8 else {}
    props = [
        float(L),
        safe_float(desc.get("net_charge_ph7"), 0.0),
        safe_float(desc.get("isoelectric_point"), 7.0),
        safe_float(desc.get("eisenberg_hydrophobic_moment"), 0.0),
        safe_float(desc.get("grand_avg_hydropathy"), 0.0),
        safe_float(desc.get("boman_index"), 0.0),
        safe_float(desc.get("instability_index"), 30.0),
    ]
    return np.array(aac + props, dtype=np.float32)

class ActivitySafetyEvaluator:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.amp_model = joblib.load(model_dir / "amp_classifier_ensemble.joblib")
        self.hemo_model = joblib.load(model_dir / "empirical_hemolysis_ensemble.joblib")

    def predict(self, seq: str) -> Dict[str, Any]:
        """Predict AMP probability, empirical hemolysis risk, and explicit uncertainties."""
        feats = extract_features(seq).reshape(1, -1)
        
        # AMP Ensemble Predictions
        tree_amp_preds = np.array([tree.predict_proba(feats)[0, 1] for tree in self.amp_model.estimators_])
        mean_amp = float(np.mean(tree_amp_preds))
        std_amp = float(np.std(tree_amp_preds))
        
        # Empirical Hemolysis Ensemble Predictions
        tree_hemo_preds = np.array([tree.predict_proba(feats)[0, 1] for tree in self.hemo_model.estimators_])
        mean_hemo = float(np.mean(tree_hemo_preds))
        std_hemo = float(np.std(tree_hemo_preds))

        # Analytical Biophysical Membrane-Disruption Proxy
        grv = feats[0, 24] # grand_avg_hydropathy
        mom = feats[0, 23] # eisenberg_moment
        biophys_proxy = float(np.clip(0.5 * (grv - 0.2) * 1.5 + 0.5 * (mom - 0.4) * 2.0, 0.0, 1.0))

        return {
            "pred_amp_probability": round(mean_amp, 4),
            "uncertainty_amp_std": round(std_amp, 4),
            "pred_empirical_hemolysis_prob": round(mean_hemo, 4),
            "uncertainty_hemo_std": round(std_hemo, 4),
            "biophysical_membrane_disruption_proxy": round(biophys_proxy, 4),
            "pred_toxicity_risk": round(0.5 * mean_hemo + 0.5 * biophys_proxy, 4),
        }
