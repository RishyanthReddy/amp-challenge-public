"""
Role 03: VAE & Latent Models — Candidate Provenance & Schema Builder
Follows Team Lead (Luna) exact candidate schema specifications.
"""

from __future__ import annotations
import json
import time
from typing import Optional, Dict, Any

CKPT_SHA256 = "b790d98ca8a053c82d56a735057a66ef833ee7721ab458cbfabeb3199f3630f9"
DECOMPOSER_SHA256 = "3ebca2d2ebcfa12f7c00e12e17ea6032223bb3943fcf36739ee4c33041ea22bf"

def build_candidate_record(
    sequence_id: str,
    sequence: str,
    run_id: str,
    generation_mode: str,
    attempt_id: str,
    random_seed: int,
    parent_sequence_id: Optional[str] = None,
    parent_sequence: Optional[str] = None,
    prototype_cluster_id: Optional[str] = None,
    amp_prob: Optional[float] = None,
    mic_prob: Optional[float] = None,
    sampling_config: Optional[dict] = None,
    condition: str = '{"c_amp": 1, "c_mic": 1}',
    latent_dim: int = 64,
    data_version: str = "1.0",
    shared_evaluator_version: str = "not_scored",
    candidate_status: str = "accepted",
) -> Dict[str, Any]:
    """
    Constructs a record matching Luna's exact candidate schema table:
    sequence_id, sequence, domain, model_name, model_version, checkpoint_sha256,
    decomposer_sha256, run_id, generation_mode, parent_sequence_id, parent_sequence,
    prototype_cluster_id, attempt_id, condition, latent_dim, sampling_config,
    random_seed, data_version, valid_standard_aa, length, shared_evaluator_version,
    candidate_status, created_utc
    """
    seq_clean = str(sequence or "").strip().upper()
    return {
        "sequence_id": sequence_id,
        "sequence": seq_clean,
        "domain": "vae",
        "model_name": "HydrAMP_epoch37",
        "model_version": "szczurek-lab/hydramp@6590d2f",
        "checkpoint_sha256": CKPT_SHA256,
        "decomposer_sha256": DECOMPOSER_SHA256,
        "run_id": run_id,
        "generation_mode": generation_mode,  # 'unconstrained' or 'analogue'
        "parent_sequence_id": parent_sequence_id if parent_sequence_id else None,
        "parent_sequence": parent_sequence if parent_sequence else None,
        "prototype_cluster_id": prototype_cluster_id if prototype_cluster_id else None,
        "attempt_id": attempt_id,
        "condition": condition,
        "latent_dim": latent_dim,
        "sampling_config": json.dumps(sampling_config or {}),
        "random_seed": random_seed,
        "data_version": data_version,
        "valid_standard_aa": True,
        "length": len(seq_clean),
        "amp_prob": round(float(amp_prob), 4) if amp_prob is not None else None,
        "mic_prob": round(float(mic_prob), 4) if mic_prob is not None else None,
        "shared_evaluator_version": shared_evaluator_version,
        "candidate_status": candidate_status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
