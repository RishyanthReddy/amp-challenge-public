from __future__ import annotations
import argparse, csv
from pathlib import Path
from .core import fasta_rows, norm, lev_ratio
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('fasta');p.add_argument('--reference',default='data/challenge/antibacterial.fasta');p.add_argument('--top100',action='store_true');a=p.parse_args(argv)
 rows,_,bad=fasta_rows(Path(a.fasta)); seq=[norm(r['sequence']) for r in rows]; refs={norm(r['sequence']) for r in fasta_rows(Path(a.reference))[0]}
 errors=[]
 if not seq:errors.append('no sequences')
 if bad:errors.append(f'malformed FASTA ({bad} sequence line(s) before a header)')
 if len(seq)!=len(set(seq)):errors.append('duplicate sequences')
 for s in seq:
  if not (8<=len(s)<=50) or set(s)-set('ACDEFGHIKLMNPQRSTVWY'):errors.append('invalid sequence');break
 if set(seq)&refs:errors.append('exact challenge-reference overlap')
 if a.top100 and any(max((lev_ratio(s,r) for r in refs),default=0)>.8 for s in seq):errors.append('top-100 similarity exceeds 80%')
 if errors: raise SystemExit('INVALID: '+ '; '.join(errors))
 print(f'VALID: {len(seq)} sequences')
if __name__=='__main__':main()
