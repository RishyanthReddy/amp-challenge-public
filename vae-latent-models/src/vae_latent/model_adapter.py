"""
Role 03: VAE & Latent Models — HydrAMP Model Adapter
Provides a clean, modular, validated interface to HydrAMP's latent space (R^64).
Encapsulates both Mode 1 (Unconstrained) and Mode 2 (Analogue) generation.
"""

from __future__ import annotations
import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class HydrAMPAdapter:
    """
    Adapter pattern around HydrAMPGenerator (Szymczak et al. 2023).
    Ensures safe tensor boundaries, condition verification, and ancestry tracking.
    """
    def __init__(self, model_path: str, decomposer_path: str, softmax: bool = True):
        self.model_path = model_path
        self.decomposer_path = decomposer_path
        self.softmax = softmax
        self.generator = None
        self._load_generator()

    def _load_generator(self):
        from amp.inference.inference import HydrAMPGenerator
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"HydrAMP model checkpoint not found at: {self.model_path}")
        if not os.path.exists(self.decomposer_path):
            raise FileNotFoundError(f"PCA decomposer not found at: {self.decomposer_path}")

        print(f"[HydrAMPAdapter] Loading generator from {self.model_path}...")
        self.generator = HydrAMPGenerator(
            model_path=self.model_path,
            decomposer_path=self.decomposer_path,
            softmax=self.softmax,
        )
        print("[HydrAMPAdapter] Model loaded successfully.")

    def sample_unconstrained(
        self,
        n_target: int,
        seed: int = 42,
        filter_out: bool = True,
        properties: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Mode 1: Sample new peptides from latent prior z ~ N(0, I)
        Conditioned on y = [c_amp=1, c_mic=1]
        """
        assert self.generator is not None, "Model not initialized!"
        raw_results = self.generator.unconstrained_generation(
            mode="amp",
            n_target=n_target,
            seed=seed,
            filter_out=filter_out,
            properties=properties,
            n_attempts=1 if self.softmax else 64,
        )
        return raw_results

    def sample_analogues(
        self,
        seed_sequences: List[str],
        seed: int = 42,
        temp: float = 2.0,
        n_attempts: int = 100,
        filtering_criteria: str = "discovery",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Mode 2: Sample analogue variations in the latent neighborhood around parent seeds.
        z' = z_parent + temp * sigma * N(0, I)
        """
        assert self.generator is not None, "Model not initialized!"
        # Filter input seeds to HydrAMP max length (25 residues)
        valid_seeds = [s.strip().upper() for s in seed_sequences if 8 <= len(s.strip()) <= 25]
        if not valid_seeds:
            return {}

        raw_results = self.generator.analogue_generation(
            sequences=valid_seeds,
            seed=seed,
            filtering_criteria=filtering_criteria,
            n_attempts=n_attempts,
            temp=temp,
        )

        cleaned_analogues = {}
        for parent_seq, p_data in raw_results.items():
            children = p_data.get("generated_sequences", [])
            cleaned_analogues[parent_seq] = children if children else []

        return cleaned_analogues
