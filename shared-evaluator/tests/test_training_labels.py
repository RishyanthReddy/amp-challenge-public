import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_evaluator_models import _similarity_groups, label_hemolysis


def assay_row(measurement, unit="µM", *, assay="50% Hemolysis", cell="Human erythrocytes", mw=2000.0):
    return pd.Series(
        {
            "assay_type": assay,
            "target_cell_type": cell,
            "measurement": measurement,
            "measurement_unit": unit,
            "molecular_weight_da": mw,
        }
    )


@pytest.mark.parametrize(
    ("measurement", "expected"),
    [
        ("99", 1),
        ("100", 0),
        ("99±2", None),
        ("<100", 1),
        ("<120", None),
        ("<=99", 1),
        ("<=100", None),
        (">100", 0),
        (">99", None),
        (">=100", 0),
    ],
)
def test_hc50_threshold_and_censoring(measurement, expected):
    assert label_hemolysis(assay_row(measurement)) == expected


def test_mass_concentration_converts_using_molecular_weight():
    # 50 µg/ml = 25 µM for a 2,000 Da peptide.
    assert label_hemolysis(assay_row("50", unit="µg/ml")) == 1


@pytest.mark.parametrize(
    "row",
    [
        assay_row("50", unit=""),
        assay_row("50", assay="25% Hemolysis"),
        assay_row("50", cell="Rabbit erythrocytes"),
        assay_row("NA"),
        assay_row("50", mw=0),
    ],
)
def test_unsupported_or_nonhuman_observations_are_unlabeled(row):
    assert label_hemolysis(row) is None


def test_similarity_groups_join_close_sequences_and_separate_distant_ones():
    groups = _similarity_groups(["AAAAAAAA", "AAAAAAAT", "YYYYYYYY"])
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
