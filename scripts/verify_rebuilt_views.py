"""Verify rebuilt curation values despite Parquet-writer and path-separator differences.

All columns, dtypes and ordered values are checked. Only source_file backslashes are
normalized to slashes. Float values use exact hexadecimal representations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def atom(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        return ["missing"]
    if isinstance(value, float):
        return ["missing"] if math.isnan(value) else ["float", value.hex()]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, str):
        return ["str", value]
    raise TypeError(f"Unsupported curated value type: {type(value).__name__}")


def fingerprint(path: Path) -> dict:
    frame = (pd.read_parquet(path) if path.suffix == ".parquet" else
             pd.read_csv(path, low_memory=False))
    if "source_file" in frame:
        frame["source_file"] = frame["source_file"].str.replace("\\", "/", regex=False)
    schema = {"columns": list(frame.columns), "dtypes": [str(x) for x in frame.dtypes],
              "rows": len(frame)}
    digest = hashlib.sha256()
    digest.update(json.dumps(schema, separators=(",", ":"), ensure_ascii=True).encode())
    digest.update(b"\n")
    for row in frame.itertuples(index=False, name=None):
        digest.update(json.dumps([atom(x) for x in row], separators=(",", ":"),
                                 ensure_ascii=True).encode())
        digest.update(b"\n")
    return {**schema, "value_sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "docs/curated_view_fingerprints.json").read_text())
    failures, checks = [], []
    for item in manifest["inputs"]:
        path = args.root.resolve() / item["path"]
        try:
            actual = fingerprint(path)
            expected = {key: item[key] for key in actual}
            if actual != expected:
                raise ValueError(f"Rebuilt curation differs: {item['path']}")
            checks.append({"path": item["path"], "rows": actual["rows"],
                           "value_sha256": actual["value_sha256"]})
        except (OSError, ValueError, TypeError) as exc:
            failures.append(str(exc))
    if failures:
        parser.exit(1, "Rebuilt input verification failed:\n" + "\n".join(failures) + "\n")
    result = {"verified_rebuilt_views": len(checks), "inputs": checks,
              "scope": manifest["scope"]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
