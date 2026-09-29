"""Exercise the root's canonical local submission verifier without a replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_submission import verify_submission


def test_canonical_submission_invariants():
    verify_submission(ROOT, replay=False)
