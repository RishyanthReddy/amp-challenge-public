from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))
from portfolio.dpp_optimizer import DppPortfolioOptimizer
from portfolio.map_elites import MapElitesArchive
from run_production_portfolio import select_apex_scored_candidates


DOMAINS = ["autoregressive", "vae_latent", "diffusion", "evolution"]


def candidate_frame(sequences):
    rows = []
    for i, (domain, sequence) in enumerate(zip(DOMAINS, sequences)):
        rows.append(
            {
                "sequence": sequence,
                "primary_domain": domain,
                "lcb_quality": 0.8 - i * 0.01,
                "net_charge_ph7": 2.0 + i,
                "eisenberg_moment": 0.2 + i * 0.1,
                "length": len(sequence),
                "isoelectric_point": 8.0 + i * 0.1,
                "gravy": -0.2 + i * 0.1,
                "boman_index": 0.2 + i * 0.1,
                "instability_index": 20.0 + i,
                "pred_amp_probability": 0.8,
                "pred_toxicity_risk": 0.2,
                "max_reference_similarity": 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_map_elites_length_edges_match_documented_residue_bins():
    archive = MapElitesArchive()
    expected = {14: 0, 15: 1, 22: 1, 23: 2, 30: 2, 31: 3, 40: 3, 41: 4, 50: 4}
    for length, index in expected.items():
        assert archive.get_cell_coordinates(3.0, 0.45, length)[2] == index


def test_dpp_selects_all_four_required_domains_and_preserves_nested_order():
    frame = candidate_frame(["ACACACAC", "DEDEDEDE", "FGFGFGFG", "HKHKHKHK"])
    result = DppPortfolioOptimizer(max_domain_quota=2).optimize_portfolio(
        frame, top_k=4, nested_k=2
    )
    assert set(result["top100_df"]["primary_domain"]) == set(DOMAINS)
    assert result["top50_df"]["sequence"].tolist() == result["top100_df"].head(2)["sequence"].tolist()


def test_dpp_supports_explicit_submission_domains_and_levenshtein_guard():
    domains = ["autoregressive", "evolution", "autoregressive", "evolution"]
    frame = candidate_frame(["ACACACAC", "DEDEDEDE", "FGFGFGFG", "HKHKHKHK"])
    frame["primary_domain"] = domains
    result = DppPortfolioOptimizer(
        required_domains={"autoregressive", "evolution"},
        max_domain_quota=2,
        max_pairwise_levenshtein=0.80,
    ).optimize_portfolio(frame, top_k=4, nested_k=2)
    assert set(result["top100_df"]["primary_domain"]) == {"autoregressive", "evolution"}
    assert len(result["top100_df"]) == 4


def test_dpp_fails_closed_when_levenshtein_guard_blocks_remaining_items():
    frame = candidate_frame(["ACDEFGHI", "ACDEFGHK", "ACDEFGHM", "ACDEFGHP"])
    frame["primary_domain"] = ["autoregressive", "evolution", "autoregressive", "evolution"]
    with pytest.raises(ValueError, match="No candidate satisfies"):
        DppPortfolioOptimizer(
            required_domains={"autoregressive", "evolution"},
            max_domain_quota=2,
            max_pairwise_cosine=1.0,
            max_pairwise_levenshtein=0.80,
        ).optimize_portfolio(frame, top_k=4, nested_k=2)


def test_dpp_fails_closed_when_no_cosine_diverse_portfolio_exists():
    sequences = ["ACDEFGHI", "CDEFGHIA", "DEFGHIAC", "EFGHIACD"]
    frame = candidate_frame(sequences)
    with pytest.raises(ValueError, match="No candidate satisfies"):
        DppPortfolioOptimizer(max_domain_quota=2).optimize_portfolio(frame, top_k=4, nested_k=2)


def test_dpp_fails_closed_when_novelty_column_is_missing():
    frame = candidate_frame(["ACACACAC", "DEDEDEDE", "FGFGFGFG", "HKHKHKHK"])
    with pytest.raises(KeyError, match="max_reference_similarity"):
        DppPortfolioOptimizer().optimize_portfolio(
            frame.drop(columns="max_reference_similarity"), top_k=4, nested_k=2
        )


def test_ranked_pool_excludes_unscored_apex_candidates():
    frame = pd.DataFrame(
        {
            "sequence": ["ACDEFGHI", "KLMNPQRS", "TVWYACDE", "FGHIKLMN"],
            "apex_mean_mic": [8.0, np.nan, np.inf, 0.0],
        }
    )
    ranked = select_apex_scored_candidates(frame, minimum=1)
    assert ranked["sequence"].tolist() == ["ACDEFGHI"]
    with pytest.raises(ValueError, match="At least 2"):
        select_apex_scored_candidates(frame, minimum=2)
    with pytest.raises(KeyError, match="apex_mean_mic"):
        select_apex_scored_candidates(frame.drop(columns="apex_mean_mic"))
