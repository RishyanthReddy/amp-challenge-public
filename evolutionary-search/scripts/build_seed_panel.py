"""
Curate Diverse Evolutionary Seed Panel (Role 05)
Selects 50 diverse, validated seeds from prototype_panel_validated.csv across
5 distinct biophysical archetypes (short cationic, amphipathic helical, Trp/Arg rich,
proline-rich/extended, and structured loop), assigning 10 family clusters.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "data-engineering/src"))
from amp_data.core import biophysical_descriptors

def categorize_archetype(seq: str, charge: float, moment: float, boman: float) -> tuple[int, str]:
    L = len(seq)
    r_w_ratio = (seq.count("R") + seq.count("W")) / L
    p_ratio = seq.count("P") / L

    if r_w_ratio >= 0.40:
        return 0, "Trp_Arg_Rich"
    elif p_ratio >= 0.15:
        return 1, "Proline_Rich_Extended"
    elif moment >= 0.65:
        return 2, "High_Amphipathic_Helical"
    elif L <= 12 and charge >= 3.0:
        return 3, "Short_Ultrashort_Cationic"
    elif moment >= 0.40:
        return 4, "Moderate_Amphipathic"
    elif boman >= 2.5:
        return 5, "High_Boman_Protein_Interacting"
    elif L >= 30:
        return 6, "Long_Membranolytic"
    elif charge >= 5.0:
        return 7, "Hyper_Cationic"
    elif seq.count("C") >= 2:
        return 8, "Cysteine_Constrained"
    else:
        return 9, "Balanced_Hydrophobic_Cationic"

def main():
    src_path = ROOT / "vae-latent-models/outputs/prototype_panel_validated.csv"
    df_raw = pd.read_csv(src_path)
    print(f"Loaded {len(df_raw)} source prototypes.")

    # Filter to 8 <= length <= 50 and 100% canonical amino acids
    canon_set = set("ACDEFGHIKLMNPQRSTVWY")
    valid_mask = df_raw["sequence"].apply(lambda s: 8 <= len(s) <= 50 and set(s).issubset(canon_set))
    df_valid = df_raw[valid_mask].copy()
    print(f"Valid length (8-50) & canonical sequences: {len(df_valid)}")

    seeds = []
    for idx, row in df_valid.iterrows():
        seq = str(row["sequence"]).strip().upper()
        chg = float(row["net_charge_ph7"])
        mom = float(row["eisenberg_moment"])
        bom = float(row["boman_index"])
        obs = int(row["activity_observation_count"])
        fam_id, arch = categorize_archetype(seq, chg, mom, bom)

        reason = (
            f"Archetype: {arch} | Length: {len(seq)} aa | Net Charge: +{chg:.1f} | "
            f"Moment: {mom:.2f} | Activity Observations: {obs} in APD/DBAASP"
        )

        seeds.append({
            "seed_id": f"seed_{len(seeds):03d}_{row['seed_id'][:12]}",
            "sequence": seq,
            "length": len(seq),
            "family_id": fam_id,
            "family_archetype": arch,
            "source_databases": row["source_databases"],
            "activity_observation_count": obs,
            "net_charge_ph7": round(chg, 3),
            "eisenberg_moment": round(mom, 3),
            "boman_index": round(bom, 3),
            "selection_reason": reason,
        })

    df_seeds = pd.DataFrame(seeds)
    # Deduplicate by sequence
    df_seeds = df_seeds.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    
    # Stratified sample: select top 50 balanced across family clusters
    balanced_seeds = []
    for fam_id, group in df_seeds.groupby("family_id"):
        # Sort by observation count descending, take up to 6 per family
        top_fam = group.sort_values(by="activity_observation_count", ascending=False).head(5)
        balanced_seeds.append(top_fam)

    final_df = pd.concat(balanced_seeds, ignore_index=True)
    if len(final_df) < 50:
        remaining = df_seeds[~df_seeds["sequence"].isin(final_df["sequence"])]
        fill = remaining.head(50 - len(final_df))
        final_df = pd.concat([final_df, fill], ignore_index=True)

    final_df = final_df.head(50).reset_index(drop=True)

    out_path = ROOT / "evolutionary-search/data/evolution_seeds.csv"
    final_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(final_df)} curated seeds across {final_df['family_id'].nunique()} families to:")
    print(f"  {out_path}")
    print("\n--- Family Archetype Breakdown ---")
    print(final_df["family_archetype"].value_counts().to_string())

if __name__ == "__main__":
    main()
