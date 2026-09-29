"""
Audit Ground-Truth Evaluator View (Role 06)
Analyzes data-engineering/data/processed/views/evaluator_view.parquet:
- Split distributions (leakage-aware train/val/test splits)
- Label distributions (is_amp, MIC observations, toxicity observations)
- Missingness patterns (missing_mic, missing_toxicity)
- Documents policy: Missing data is strictly "unknown", never treated as "safe"
Generates reports/evaluator_view_audit.md
"""

import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(__file__).resolve().parents[2]

def main():
    print("=======================================================")
    print(" Auditing Ground-Truth Evaluator View (Role 06)")
    print("=======================================================\n")

    view_path = MAIN_ROOT / "data-engineering/data/processed/views/evaluator_view.parquet"
    if not view_path.exists():
        raise FileNotFoundError(f"{view_path} not found!")

    df = pd.read_parquet(view_path)
    print(f"Loaded {len(df)} rows from {view_path.name}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}\n")

    # 1. Split Distribution
    split_counts = df["split"].value_counts().to_dict()
    print("--- Split Breakdown ---")
    for s, c in split_counts.items():
        print(f"  {s}: {c} ({c/len(df)*100:.1f}%)")

    # 2. Activity & Ground Truth Labels
    amp_counts = df["is_amp"].value_counts().to_dict()
    has_mic = (df["mic_observation_count"] > 0).sum()
    has_tox = (df["toxicity_observation_count"] > 0).sum()
    missing_mic = df["missing_mic"].sum()
    missing_tox = df["missing_toxicity"].sum()

    print(f"\n--- Ground Truth Labels ---")
    print(f"  is_amp == True:  {amp_counts.get(True, 0)}")
    print(f"  is_amp == False: {amp_counts.get(False, 0)}")
    print(f"  With MIC observations:      {has_mic} ({has_mic/len(df)*100:.1f}%)")
    print(f"  Missing MIC data:           {missing_mic} ({missing_mic/len(df)*100:.1f}%)")
    print(f"  With Toxicity observations: {has_tox} ({has_tox/len(df)*100:.1f}%)")
    print(f"  Missing Toxicity data:      {missing_tox} ({missing_tox/len(df)*100:.1f}%)")

    # 3. Organisms & Strains coverage
    org_counts = df["organisms"].dropna().nunique()
    print(f"\nDistinct tested organisms: {org_counts}")

    # 4. Generate Markdown Audit Report
    report_md = f"""# Role 06: Ground-Truth Evaluator View Audit Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Source Dataset:** `data-engineering/data/processed/views/evaluator_view.parquet`  
**Data Version:** 1.0 (Frozen by Role 01 Data Engineering)  
**Total Records:** {len(df)}  

---

## 1. Split & Cluster Partitioning

The master dataset partitions sequences to prevent data leakage between sequence families:

| Split | Sequence Count | Percentage | Purpose |
| :--- | :--- | :--- | :--- |
| **core_train_only** | {split_counts.get('core_train_only', 0)} | {split_counts.get('core_train_only', 0)/len(df)*100:.1f}% | Training set for ML classifiers & predictors |
| **train** | {split_counts.get('train', 0)} | {split_counts.get('train', 0)/len(df)*100:.1f}% | General training split across diverse clusters |
| **val** | {split_counts.get('val', 0)} | {split_counts.get('val', 0)/len(df)*100:.1f}% | Hyperparameter tuning & model selection |
| **test** | {split_counts.get('test', 0)} | {split_counts.get('test', 0)/len(df)*100:.1f}% | Final held-out evaluation & calibration |

---

## 2. Evidence Labels & Missingness Summary

| Label / Dimension | Present Count | Missing / Unknown | Missingness Policy |
| :--- | :--- | :--- | :--- |
| **Antimicrobial Status (`is_amp`)** | {amp_counts.get(True, 0)} True, {amp_counts.get(False, 0)} False | {len(df) - sum(amp_counts.values())} unknown | Only experimentally confirmed AMPs are labeled positive. |
| **MIC Potency Measurements** | {has_mic} sequences | {missing_mic} sequences ({missing_mic/len(df)*100:.1f}%) | Missing MIC is explicitly tracked; not imputed as inactive. |
| **Hemolysis / Toxicity Observations** | {has_tox} sequences | {missing_tox} sequences ({missing_tox/len(df)*100:.1f}%) | **Missing toxicity is STRICTLY UNKNOWN, never treated as safe.** |

---

## 3. Evaluator Policy Directives
1. **No Leakage:** Evaluator models must be trained strictly on `core_train_only` and `train` splits. Validation and testing must use strictly held-out `val` and `test` clusters.
2. **Missingness Integrity:** When predicting safety or hemolysis, candidates with missing ground-truth or predictor uncertainty must be flagged as `unknown_safety` rather than awarded favorable scores.
"""

    rep_path = ROOT / "reports/evaluator_view_audit.md"
    with open(rep_path, "w") as f:
        f.write(report_md)
    print(f"✓ Saved evaluator view audit report to {rep_path}")

if __name__ == "__main__":
    main()
