"""Compatibility wrapper for the repository's single submission packager.

Run from any working directory. The canonical implementation and output paths live in
``src/amp_challenge_2027/generate.py`` at the repository root.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from amp_challenge_2027.generate import main


if __name__ == "__main__":
    main()
