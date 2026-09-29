"""Quantify unresolved source dependencies without changing training data or assets."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data-engineering/src'))
from amp_data.core import xlsx_rows


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    view = ROOT / 'data-engineering/data/processed/views/autoregressive_view.csv'
    provenance = ROOT / 'data-engineering/data/processed/provenance.csv'
    workbook = ROOT / 'data-engineering/data/raw/DRAMP/Antibacterial_amps.xlsx'
    train = [r for r in read_csv(view) if r['split'] in {'train', 'core_train_only'}]
    sources = defaultdict(set)
    for row in read_csv(provenance):
        sources[row['sequence_id']].add(row['source_database'])
    patent_sequences = set()
    categories = Counter()
    for _, rows, _ in xlsx_rows(workbook):
        for row in rows:
            category = str(row.get('Dataset', '')).strip()
            categories[category] += 1
            if category.casefold() == 'patent':
                patent_sequences.add(''.join(str(row['Sequence']).split()).upper())
    if not train or not categories or not patent_sequences:
        raise ValueError('Missing training or patent-source records; audit cannot pass silently')
    counts = Counter()
    no_provenance = 0
    for row in train:
        src = sources[row['sequence_id']]
        no_provenance += not bool(src)
        counts.update(src)
    report = {
        'scope': 'Current training view source overlap; not proof of historical training bytes or legal clearance',
        'training_rows': len(train),
        'training_unique_sequences': len({r['sequence'] for r in train}),
        'missing_provenance_rows': no_provenance,
        'source_overlap_counts_nonexclusive': dict(sorted(counts.items())),
        'dramp_antibacterial_categories': dict(sorted(categories.items())),
        'training_sequences_matching_dramp_patent_records': len({r['sequence'] for r in train} & patent_sequences),
        'inputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (view, provenance, workbook)},
        'rights_status': 'UNRESOLVED',
        'remediation': 'Obtain applicable permissions, or retrain generators and evaluators from a documented permitted corpus and regenerate/reselect artifacts. Removing raw files alone does not remove model training influence.',
    }
    out = ROOT / 'docs/training_source_scope.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if no_provenance:
        raise ValueError('Training records without provenance')


if __name__ == '__main__':
    main()
