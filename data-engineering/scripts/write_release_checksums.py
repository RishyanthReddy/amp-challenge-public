"""Write deterministic SHA-256 checksums for the versioned data handover."""
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (ROOT / "data" / "processed", ROOT / "data" / "reports")
EXCLUDE = {"SHA256SUMS"}

def main() -> None:
    files = sorted(p for folder in INCLUDE for p in folder.rglob("*") if p.is_file() and p.name not in EXCLUDE)
    lines = []
    for path in files:
        digest = sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "data" / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
