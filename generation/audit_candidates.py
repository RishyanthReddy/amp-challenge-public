"""Audit freshly generated AR/evolution candidates against the public reference."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data-engineering/src'))
from amp_data.core import fasta_rows, norm, lev_ratio


def main() -> None:
    reference = ROOT/'data-engineering/data/challenge/antibacterial.fasta'
    rows, _, malformed = fasta_rows(reference)
    refs = sorted({norm(r['sequence']) for r in rows})
    if malformed or len(rows) != 39448 or len(refs) != 39448:
        raise ValueError('The reference must contain 39,448 valid unique peptides.')
    lengths = {}
    for ref in refs:
        lengths.setdefault(len(ref),[]).append(ref)
    for name in ['ar','evolution']:
        path = ROOT/f'outputs/{name}_candidates.csv'
        df = pd.read_csv(path)
        results = []
        for seq in df['sequence']:
            seq = norm(seq)
            targets = [ref for length in range(math.ceil(len(seq)*2/3),math.floor(len(seq)*1.5)+1) for ref in lengths.get(length,[])]
            results.append(max((lev_ratio(seq,ref) for ref in targets),default=0.0))
        df['max_reference_similarity'] = results
        df['passes_novelty_rule_le80'] = df['max_reference_similarity'] <= 0.80
        df = df.loc[~df['sequence'].map(norm).isin(set(refs))].reset_index(drop=True)
        df.to_csv(path,index=False)
        print(f'Audited {name}: {len(df)} candidates',flush=True)
    context = ROOT/'generation/comparison_pool_manifest.json'
    manifest = json.loads(context.read_text())
    for filename, record in manifest['files'].items():
        source = ROOT/'generation'/filename
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Baseline comparison input was changed: {filename}')
        df = pd.read_csv(source)
        if len(df) != record['rows']:
            raise ValueError(f'Baseline comparison row count differs: {filename}')
        name = 'vae' if filename.startswith('vae') else 'diffusion'
        df.to_csv(ROOT/f'outputs/{name}_candidates.csv',index=False)


if __name__ == '__main__':
    main()
