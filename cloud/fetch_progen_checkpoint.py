"""Fetch and verify the pinned ProGen2 checkpoint from public GitHub, GitHub CLI, or Beam."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "autoregressive-models/checkpoints/progen2_small_amp_best_val"
REMOTE_DIRECTORY = "beam://amp-models/progen2_small_amp_best_val"

# SHA-256 and byte size captured from the Beam volume used for the Role 02 run.
ASSETS = {
    "config.json": (
        1_206,
        "44decbdd5309ba39ea252e4e831779f305233c3aed391badc8d586188e167d4a",
    ),
    "generation_config.json": (
        111,
        "7be49f20354844a419c088af9d14248444773e3592ea4da77f231824ba1003c8",
    ),
    "model.safetensors": (
        302_307_720,
        "124b8ea7df5c96cda927bada51c9d26d89f91636d0975fe70e7bc0ef182f9f83",
    ),
    "tokenizer.json": (
        1_756,
        "cc489cd8bfeab3c70c6a2954d2963b9fb8e7ee4b13aaa0e84e97d7f17f35d43c",
    ),
}
GITHUB_REPOSITORY = "RishyanthReddy/amp-challenge-public"
GITHUB_TAG = "progen2-checkpoint-20260928"
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, name: str) -> bool:
    expected_size, expected_sha256 = ASSETS[name]
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and sha256_file(path) == expected_sha256
    )


def fetch(destination: Path, beam_cli: str = "beam", *, source: str = "github",
          github_repository: str = GITHUB_REPOSITORY, gh_cli: str = "gh") -> None:
    if source not in {"public", "github", "beam"}:
        raise ValueError(f"Unknown checkpoint source: {source}")
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check existing files before contacting Beam. Never overwrite an unknown or
    # mismatching local checkpoint, and fetch into a staging directory first.
    for name in ASSETS:
        local_path = destination / name
        if local_path.exists() and not verify_asset(local_path, name):
            raise ValueError(
                f"Refusing to overwrite an asset with unexpected size/hash: {local_path}"
            )

    with tempfile.TemporaryDirectory(
        prefix=".progen2-download-", dir=destination.parent
    ) as tmp:
        staging = Path(tmp)
        for name in ASSETS:
            if (destination / name).is_file():
                print(f"Already verified: {destination / name}")
                continue

            staged_path = staging / name
            remote_path = f"{REMOTE_DIRECTORY}/{name}"
            if source == "beam":
                subprocess.run([beam_cli, "cp", remote_path, name], check=True, cwd=staging)
            elif source == "public":
                url = f"https://github.com/{github_repository}/releases/download/{GITHUB_TAG}/{name}"
                with urllib.request.urlopen(url, timeout=60) as response, staged_path.open("wb") as output:
                    downloaded = 0
                    for block in iter(lambda: response.read(CHUNK_SIZE), b""):
                        downloaded += len(block)
                        if downloaded > ASSETS[name][0]:
                            raise ValueError(f"Downloaded asset exceeds its pinned size: {name}")
                        output.write(block)
            else:
                subprocess.run(
                    [gh_cli, "release", "download", GITHUB_TAG, "--repo", github_repository,
                     "--pattern", name, "--dir", str(staging)], check=True,
                )
            if not verify_asset(staged_path, name):
                actual_size = staged_path.stat().st_size if staged_path.exists() else 0
                actual_sha256 = sha256_file(staged_path) if staged_path.is_file() else "missing"
                raise ValueError(
                    f"Downloaded ProGen2 asset did not match its pin: {name}; "
                    f"got {actual_size} bytes / {actual_sha256}"
                )

        # Install only after every missing file has been downloaded and verified.
        destination.mkdir(parents=True, exist_ok=True)
        for name in ASSETS:
            staged_path = staging / name
            local_path = destination / name
            if staged_path.is_file() and not local_path.exists():
                os.replace(staged_path, local_path)

    missing_or_invalid = [
        name for name in ASSETS if not verify_asset(destination / name, name)
    ]
    if missing_or_invalid:
        raise RuntimeError(f"ProGen2 checkpoint incomplete after fetch: {missing_or_invalid}")
    print(f"Verified ProGen2 checkpoint cache: {destination}")
    print(f"model.safetensors SHA-256: {ASSETS['model.safetensors'][1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--beam-cli", default="beam", help="Beam CLI executable")
    parser.add_argument("--source", choices=("public", "github", "beam"), default="github")
    parser.add_argument("--github-repository", default=GITHUB_REPOSITORY)
    parser.add_argument("--gh-cli", default="gh")
    args = parser.parse_args()
    fetch(args.output, args.beam_cli, source=args.source,
          github_repository=args.github_repository, gh_cli=args.gh_cli)


if __name__ == "__main__":
    main()
