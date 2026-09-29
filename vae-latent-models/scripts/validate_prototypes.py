"""
Role 03: Validate Prototype Seed Panel for HydrAMP Analogue Generation
Filters and documents the 100-seed panel from Role 01 data engineering.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_PANEL = ROOT / "data-engineering/data/processed/curated_seed_panel.csv"
OUT_DIR = ROOT / "vae-latent-models/outputs"
DOCS_DIR = ROOT / "vae-latent-models/docs"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading seed panel from {IN_PANEL}...")
    df = pd.read_csv(IN_PANEL)
    print(f"Loaded {len(df)} total candidate seeds.")

    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

    records = []
    for _, row in df.iterrows():
        seq = str(row["sequence"]).strip().upper()
        L = len(seq)
        is_canonical = set(seq).issubset(STANDARD_AAS)
        length_ok = 8 <= L <= 25  # HydrAMP hard input shape limit is 25

        status = "eligible" if (is_canonical and length_ok) else (
            "incompatible_length_gt25" if L > 25 else "invalid_residues"
        )

        records.append({
            "seed_id": row["sequence_id"],
            "sequence": seq,
            "length": L,
            "cluster_id": row.get("cluster_id", "unassigned"),
            "source_databases": row.get("source_databases", "unknown"),
            "activity_observation_count": row.get("activity_observation_count", 0),
            "net_charge_ph7": row.get("net_charge_ph7", 0.0),
            "eisenberg_moment": row.get("eisenberg_hydrophobic_moment", 0.0),
            "boman_index": row.get("boman_index", 0.0),
            "hydramp_eligible": status == "eligible",
            "eligibility_status": status,
        })

    out_df = pd.DataFrame(records)
    out_csv = OUT_DIR / "prototype_panel_validated.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"✓ Saved validated prototype panel to {out_csv}")

    n_eligible = out_df["hydramp_eligible"].sum()
    n_clusters = out_df[out_df["hydramp_eligible"]]["cluster_id"].nunique()
    print(f"HydrAMP-eligible seeds (length <= 25): {n_eligible} across {n_clusters} clusters.")

    # Write prototype selection documentation
    doc_md = f"""# Prototype Selection & HydrAMP Length Scope Report

**Source Panel:** `data-engineering/data/processed/curated_seed_panel.csv` (100 seeds)
**HydrAMP Hardware Dimension:** `input_shape: [25, 21]` (Max 25 residues)
**Validated Output:** `outputs/prototype_panel_validated.csv`

## Summary of Seed Eligibility
- Total Seeds Audited: {len(out_df)}
- **HydrAMP-Eligible (8 <= L <= 25 residues):** **{n_eligible} seeds**
- **Incompatible Length (L > 25 residues):** **{len(out_df) - n_eligible} seeds** (flagged and excluded from VAE analogue perturbation to prevent arbitrary truncation)
- **Represented Sequence Clusters:** **{n_clusters} independent clusters**

## Cluster Distribution of Eligible Seeds
```
{out_df[out_df['hydramp_eligible']]['cluster_id'].value_counts().to_string()}
```

## Quota Balancing Directive
For Mode 2 (Analogue Generation), attempts are distributed uniformly across the **{n_eligible} eligible seeds** ({n_clusters} clusters), ensuring balanced coverage without single-family collapse.
"""
    doc_path = DOCS_DIR / "prototype_selection.md"
    with open(doc_path, "w") as f:
        f.write(doc_md)
    print(f"✓ Saved prototype documentation to {doc_path}")

if __name__ == "__main__":
    main()
