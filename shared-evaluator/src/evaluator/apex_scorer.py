"""
APEX Multi-Strain Scorer Wrapper (Role 06)
Wraps the official APEX-pathogen ensemble (diffusion-models/apex/APEX_predict.py):
- Predicts individual MICs (uM) against the 11 clinical pathogen panel:
  1. A. baumannii ATCC 19606
  2. E. coli ATCC 11775
  3. E. coli AIC221
  4. E. coli AIC222
  5. K. pneumoniae ATCC 13883
  6. P. aeruginosa PA01
  7. P. aeruginosa PA14
  8. S. aureus ATCC 12600
  9. S. aureus (ATCC BAA-1556) - MRSA
  10. vancomycin-resistant E. faecalis ATCC 700802
  11. vancomycin-resistant E. faecium ATCC 700221
- Computes broad-spectrum mean MIC and lowest-MIC (peak potency)
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

APEX_DIR = Path(__file__).resolve().parents[3] / "diffusion-models/apex"

PATHOGEN_PANEL = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]

class ApexScorer:
    def __init__(self, apex_dir: Path = APEX_DIR):
        self.apex_dir = apex_dir
        if not (self.apex_dir / "APEX_predict.py").exists():
            raise FileNotFoundError(f"APEX_predict.py missing in {self.apex_dir}")

    def score_batch(self, sequences: List[str], batch_size: int = 1000) -> pd.DataFrame:
        """
        Score a list of sequences across the 11-pathogen APEX panel in batches.
        Returns a DataFrame indexed by sequence with columns for each pathogen + mean/min MIC.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        clean_seqs = [str(s).strip().upper() for s in sequences if 8 <= len(str(s).strip()) <= 50]
        unique_seqs = sorted(list(set(clean_seqs)))

        if not unique_seqs:
            return pd.DataFrame(columns=PATHOGEN_PANEL + ["apex_mean_mic", "apex_min_mic"])

        batches = []
        for offset in range(0, len(unique_seqs), batch_size):
            sequence_batch = unique_seqs[offset:offset + batch_size]
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                fasta_file = tmp_path / "input.fasta"
                out_csv = tmp_path / "output.csv"

                with fasta_file.open("w") as handle:
                    for idx, seq in enumerate(sequence_batch):
                        handle.write(f">seq_{idx}\n{seq}\n")

                cmd = [
                    "uv", "run", "python", "APEX_predict.py",
                    "-i", str(fasta_file),
                    "-o", str(out_csv),
                    "-g", "0",
                ]
                result = subprocess.run(
                    cmd,
                    cwd=str(self.apex_dir),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"APEX scoring failed: {result.stderr}")

                batch_result = pd.read_csv(out_csv, index_col=0)
                missing_columns = set(PATHOGEN_PANEL) - set(batch_result.columns)
                if missing_columns:
                    raise RuntimeError(f"APEX output is missing pathogen columns: {sorted(missing_columns)}")
                if batch_result.index.has_duplicates or set(batch_result.index.astype(str)) != set(sequence_batch):
                    raise RuntimeError("APEX output sequence rows do not match the submitted batch")
                batch_result[PATHOGEN_PANEL] = batch_result[PATHOGEN_PANEL].apply(
                    pd.to_numeric, errors="coerce"
                )
                if not np.isfinite(batch_result[PATHOGEN_PANEL].to_numpy(dtype=float)).all():
                    raise RuntimeError("APEX output contains non-finite pathogen scores")
                batch_result["apex_mean_mic"] = batch_result[PATHOGEN_PANEL].mean(axis=1).round(3)
                batch_result["apex_min_mic"] = batch_result[PATHOGEN_PANEL].min(axis=1).round(3)
                batches.append(batch_result)

        result = pd.concat(batches, axis=0)
        if result.index.has_duplicates or set(result.index.astype(str)) != set(unique_seqs):
            raise RuntimeError("Batched APEX result does not cover each unique sequence exactly once")
        return result
