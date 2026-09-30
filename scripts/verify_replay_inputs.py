"""Verify the exact curated inputs required for historical training and evolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fetch_training_sources import destination, verify

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((ROOT / "docs/replay_inputs.json").read_text())
    failures = []
    for item in manifest["inputs"]:
        try:
            verify(destination(root, item["path"]), dict(item, id=item["path"]))
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    if failures:
        parser.exit(1, "Archived replay inputs are incomplete or different:\n" + "\n".join(failures) + "\n")
    print(json.dumps({"verified_curated_inputs": len(manifest["inputs"]),
                      "scope": manifest["scope"]}, indent=2))


if __name__ == "__main__":
    main()
