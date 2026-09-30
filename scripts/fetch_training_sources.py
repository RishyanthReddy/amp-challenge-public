"""Restore pinned public training inputs; never substitute a newer database export.

The manifest describes retrieval and integrity, not a new license grant. Read
docs/DATA_ACCESS.md before using third-party inputs. Unknown historical acquisition
dates remain unknown even when an identical file is found online.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/training_sources.json"


def fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def destination(root: Path, relative: str) -> Path:
    path = root / relative
    resolved = path.resolve()
    if Path(relative).is_absolute() or resolved == root or root not in resolved.parents:
        raise ValueError(f"Input path escapes the destination root: {relative}")
    return resolved


def read_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported training-source manifest schema")
    entries = payload["inputs"]
    ids, paths = set(), set()
    for item in entries:
        if item["id"] in ids or item["path"] in paths:
            raise ValueError("Duplicate source ID or destination path in manifest")
        ids.add(item["id"])
        paths.add(item["path"])
        if not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError(f"Invalid SHA-256 for {item['id']}")
        if not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise ValueError(f"Invalid byte count for {item['id']}")
        for url in item["download_urls"]:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"Only HTTPS downloads are allowed: {item['id']}")
    return entries


def verify(path: Path, item: dict) -> None:
    actual = fingerprint(path)
    expected = item["bytes"], item["sha256"]
    if actual != expected:
        raise ValueError(
            f"Source mismatch for {item['id']}: got {actual[0]} bytes / {actual[1]}; "
            f"expected {expected[0]} / {expected[1]}. File was not replaced."
        )


def download(item: dict, path: Path, timeout: float) -> None:
    failures = []
    for url in item["download_urls"]:
        try:
            request = Request(url, headers={"User-Agent": "AMP-Challenge-source-verifier/1"})
            with urlopen(request, timeout=timeout) as response, path.open("wb") as stream:
                if urlparse(response.geturl()).scheme != "https":
                    raise ValueError("Download redirected to a non-HTTPS URL")
                received = 0
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > item["bytes"]:
                        raise ValueError("Download exceeds pinned file size")
                    stream.write(block)
            verify(path, item)
            return
        except (OSError, ValueError, URLError) as exc:
            failures.append(f"{url}: {exc}")
    raise RuntimeError(f"No verified download for {item['id']}: " + "; ".join(failures))


def restore(root: Path, entries: list[dict], *, verify_only: bool, timeout: float) -> dict:
    root = root.expanduser().resolve()
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    paths = {item["id"]: destination(root, item["path"]) for item in entries}
    missing = []
    for item in entries:
        path = paths[item["id"]]
        if path.exists():
            verify(path, item)
        else:
            missing.append(item)
    unavailable = [item["id"] for item in missing if verify_only or not item["download_urls"]]
    if unavailable:
        raise FileNotFoundError(
            "Exact inputs unavailable: " + ", ".join(unavailable)
            + ". Restore retained files with their recorded hashes; a current export is not a substitute."
        )

    # Validate all caches first, then download all missing inputs before installing any.
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".training-source-download-", dir=root) as tmp:
        staging = Path(tmp)
        for index, item in enumerate(missing):
            download(item, staging / str(index), timeout)
        # Recheck before installation: do not replace a file created during download.
        for item in entries:
            path = paths[item["id"]]
            if path.exists():
                verify(path, item)
        for index, item in enumerate(missing):
            target = paths[item["id"]]
            target.parent.mkdir(parents=True, exist_ok=True)
            # Atomic, exclusive installation avoids partial files and overwrites.
            try:
                os.link(staging / str(index), target)
            except FileExistsError:
                verify(target, item)
            verify(target, item)
    return {"verified_inputs": len(entries), "downloaded_inputs": len(missing),
            "inputs": [{"id": item["id"], "path": item["path"], "sha256": item["sha256"]}
                       for item in entries]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--available-only", action="store_true",
                        help="Select inputs bundled here or with exact download URLs")
    parser.add_argument("--id", action="append", help="Select individual manifest input IDs")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        all_entries = read_manifest(args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Invalid training-source manifest: {exc}\n")
    entries = all_entries
    if args.id:
        unknown = set(args.id) - {item["id"] for item in entries}
        if unknown:
            parser.error(f"Unknown input IDs: {sorted(unknown)}")
        entries = [item for item in entries if item["id"] in args.id]
    if args.available_only:
        entries = [item for item in entries if item["bundled"] or item["download_urls"]]
    if not entries:
        parser.error("No inputs selected")
    try:
        report = restore(args.root, entries, verify_only=args.verify_only, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Training input verification failed: {exc}\n")
    report["omitted_inputs"] = [item["id"] for item in all_entries if item not in entries]
    report["complete_training_source_inventory_verified"] = len(entries) == len(all_entries)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
