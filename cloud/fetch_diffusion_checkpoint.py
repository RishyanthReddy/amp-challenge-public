"""Fetch the pinned AMP-Diffusion checkpoint from its official starter-kit release."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "cloud/diffusion_checkpoint/model.pt"
SOURCE_COMMIT = "1a862af9078e6b55c87d1fa576f3da81851ba94b"
SOURCE_URL = (
    "https://media.githubusercontent.com/media/szczurek-lab/ampdiffusion-starter-kit/"
    f"{SOURCE_COMMIT}/checkpoint/model.pt"
)
EXPECTED_SHA256 = "6a3f347df7c02ff6008ac3d2d4826daeadf7418cb6862a6599b4f710e1d7f8aa"
EXPECTED_SIZE = 132_526_179
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != EXPECTED_SIZE:
        return False
    return sha256_file(path) == EXPECTED_SHA256


def fetch(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if verify(destination):
            print(f"Checkpoint already verified: {destination}")
            return
        raise ValueError(
            f"Refusing to overwrite existing file with an unexpected size/hash: {destination}"
        )

    temporary_path: Path | None = None
    try:
        request = Request(SOURCE_URL, headers={"User-Agent": "amp-challenge-checkpoint-fetch/1"})
        digest = hashlib.sha256()
        downloaded = 0
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".model.pt.", suffix=".download", dir=destination.parent, delete=False
            ) as output:
                temporary_path = Path(output.name)
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)

        actual_sha = digest.hexdigest()
        if downloaded != EXPECTED_SIZE or actual_sha != EXPECTED_SHA256:
            raise ValueError(
                f"Checkpoint mismatch: got {downloaded} bytes / SHA-256 {actual_sha}; "
                f"expected {EXPECTED_SIZE} bytes / {EXPECTED_SHA256}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Downloaded and verified checkpoint: {destination}")
        print(f"SHA-256: {EXPECTED_SHA256}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    fetch(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
