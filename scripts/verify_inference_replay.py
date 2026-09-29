"""Fetch pinned weights and compare two seeded CPU inference runs (not a 50k replay)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/inference_replay_check')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)
    env.pop('UV_PROJECT_ENVIRONMENT', None)
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    def run(command):
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    run([sys.executable, str(ROOT / 'cloud/fetch_progen_checkpoint.py')])
    checkpoint = ROOT / 'autoregressive-models/checkpoints/progen2_small_amp_best_val'
    hashes = []
    for tag in ('first', 'second'):
        destination = output / tag
        if destination.exists():
            raise FileExistsError(f'Use a fresh output directory: {destination}')
        run(['uv', 'run', '--project', 'autoregressive-models', '--locked', 'python',
             'autoregressive-models/scripts/06_generate_from_checkpoint.py',
             '--checkpoint', str(checkpoint), '--n', '1', '--seed', '42',
             '--device', 'cpu', '--max-perplexity', '100', '--output-dir', str(destination)])
        path = destination / 'finetuned_candidates.csv'
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0]['valid'] != 'True':
            raise ValueError('Inference did not return exactly one valid candidate')
        ppl = float(rows[0]['perplexity'])
        if not math.isfinite(ppl) or ppl > 100:
            raise ValueError('Perplexity screening failed')
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if hashes[0] != hashes[1]:
        raise ValueError('Seeded inference outputs differ')
    report = {'status': 'PASS', 'scope': 'two one-sequence CPU inference runs; not full portfolio replay',
              'seed': 42, 'candidate_csv_sha256': hashes,
              'checkpoint_source': 'authenticated private GitHub release',
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
