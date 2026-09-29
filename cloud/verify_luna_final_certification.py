"""Compatibility command for current local artifact verification.

This legacy filename does not issue a production, biological, or competition-eligibility
certificate. It runs the current Role 02–07 physical verifiers and the strict FASTA replay
check, then reports only whether those local checks completed successfully.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    checks = [
        ROOT / "cloud/verify_role2_physical.py",
        ROOT / "cloud/verify_role3_physical.py",
        ROOT / "cloud/verify_role4_physical.py",
        ROOT / "cloud/verify_role5_physical.py",
        ROOT / "cloud/verify_role6_physical.py",
        ROOT / "cloud/verify_role7_physical.py",
    ]
    for check in checks:
        print(f"\n--- {check.relative_to(ROOT)} ---", flush=True)
        subprocess.run([sys.executable, str(check)], cwd=ROOT, check=True)

    print("\n--- scripts/verify_submission.py (including replay) ---", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_submission.py"), str(ROOT)],
        cwd=ROOT,
        check=True,
    )
    print(
        "\nLocal Role 02–07 manifests and FASTA packaging checks passed. "
        "Competition eligibility, source-data rights, and clean-clone model access remain "
        "unresolved; this is not a production-readiness certificate."
    )


if __name__ == "__main__":
    main()
