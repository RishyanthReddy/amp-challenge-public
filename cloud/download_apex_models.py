"""
Download and verify APEX pathogen models from GitHub media server
"""

import hashlib
import os
import urllib.parse
from pathlib import Path

APEX_DIR = Path(__file__).resolve().parents[1] / "diffusion-models/apex/APEX_pathogen_models"

MODELS = [
    ("APEX_2&256&2048&1e-05&0.0&0.1", "fccd356cbb901c08fe7d35663f3902a5d2d01a4c257203af931aabf02bdd7913"),
    ("APEX_3&128&1024&1e-06&0.001&2.0", "857e17d7703d771f82fe63dd82f15a86dd60a0b4f705577a56b88485d60a60c6"),
    ("APEX_3&128&2048&1e-06&0.0&1.0", "9db48419381d7bcdac2ad3154aa6b67327c6da47dce973bb8a0745e5632ff79a"),
    ("APEX_3&128&2048&1e-06&0.001&0.1", "9362c006ccc0e7ae3fd5210f2cbbe9ed02cec4ff6383f95a119acab9b020aea4"),
    ("APEX_3&128&2048&1e-06&0.001&2.0", "58b5eeb6c8c8c4b94d1d776c92f5c653177cfcf3b77ec41cf409839304c8cc80"),
    ("APEX_3&256&2048&1e-05&0.001&0.1", "6d3f2bd66c6b3f138e98fb13d1a3010ee94714b9ebf9673475192b365d388858"),
    ("APEX_3&256&2048&1e-06&0.001&0.1", "69f101c4e3a1309166493f96221a8c44e37c87bd7a988240b32e26819162c954"),
    ("APEX_3&256&512&1e-06&0.0&2.0", "d56b8153a4618620f3b3f91aeb204d6ba8ac38af2ab2f3dff77a28863bc49059"),
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    base_url = "https://media.githubusercontent.com/media/szczurek-lab/ampdiffusion-starter-kit/main/apex/APEX_pathogen_models/"
    for fname, expected_hash in MODELS:
        target_path = APEX_DIR / fname
        if target_path.exists() and target_path.stat().st_size > 1000:
            cur_sha = sha256_file(target_path)
            if cur_sha == expected_hash:
                print(f"✓ {fname} already verified! ({target_path.stat().st_size/1e6:.1f} MB)")
                continue

        url = base_url + urllib.parse.quote(fname)
        print(f"Downloading {fname} from {url}...")
        ret = os.system(f'curl -L -s -o "{target_path}" "{url}"')
        cur_sha = sha256_file(target_path)
        print(f"  Downloaded: {target_path.stat().st_size / 1e6:.1f} MB | SHA-256 match: {cur_sha == expected_hash}")

if __name__ == "__main__":
    main()
