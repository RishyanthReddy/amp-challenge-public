import argparse, importlib.util, hashlib, json
import importlib.metadata
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
parser=argparse.ArgumentParser(description='Disclose evaluator membership without fitting models.')
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args=parser.parse_args()
root=args.source_root.resolve()
out=args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
script=root/'shared-evaluator/scripts/train_evaluator_models.py'
spec=importlib.util.spec_from_file_location('training_membership',script)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
h=lambda s:hashlib.sha256(s.encode()).hexdigest()
view=pd.read_parquet(root/'data-engineering/data/processed/views/evaluator_view.parquet')
pos=view[view.is_amp.astype(bool)]
parts={'train':['train','core_train_only'],'validation':['validation'],'test':['test']}
positive={k:m._valid_sequences(pos.loc[pos.split.isin(v),'sequence']) for k,v in parts.items()}
all_positive=set(sum(positive.values(),[]))
rows=[]; sources=[]; selected={}; inputs=[script,root/'data-engineering/data/processed/views/evaluator_view.parquet']
prov=pd.read_csv(root/'data-engineering/data/processed/provenance.csv',keep_default_na=False)
prov['sequence']=prov.sequence_normalized.str.strip().str.upper()
for k,seed,suffix in [('train',42,'train'),('validation',43,'val'),('test',44,'test')]:
 p=root/f'vae-latent-models/data/training/Uniprot_0_25_{suffix}.csv'; inputs.append(p)
 neg=[s for s in m._load_uniprot_split(p) if s not in all_positive]
 seqs,labels=m._balanced_pairs(positive[k],neg,seed); selected[k]=set(seqs)
 for i,(s,y) in enumerate(zip(seqs,labels)):
  rows.append(dict(partition=k,partition_row=i,sequence_sha256=h(s),label=int(y)))
 selected_pos={s for s,y in zip(seqs,labels) if y==1}
 for r in prov[prov.sequence.isin(selected_pos)].itertuples():
  sources.append(dict(partition=k,sequence_sha256=h(r.sequence),source_database=r.source_database,source_file=r.source_file,source_record_id=r.source_record_id,source_row=''))
 frame=pd.read_csv(p,keep_default_na=False)
 for i,r in frame.iterrows():
  s=str(r['Sequence']).strip().upper()
  if s in selected[k] and s not in selected_pos:
   sources.append(dict(partition=k,sequence_sha256=h(s),source_database='UniProt-derived HydrAMP partition',source_file=str(p.relative_to(root)),source_record_id=r['Name'],source_row=i))
assert not(selected['train']&selected['validation'] or selected['train']&selected['test'] or selected['validation']&selected['test'])
assert {r['sequence_sha256'] for r in rows}=={r['sequence_sha256'] for r in sources}
pd.DataFrame(rows).to_csv(out/'EVALUATOR_AMP_MEMBERSHIP.csv',index=False)
pd.DataFrame(sources).drop_duplicates().sort_values(['partition','sequence_sha256','source_database','source_record_id']).to_csv(out/'EVALUATOR_AMP_SOURCE_IDS.csv',index=False)
rbc,stats=m._load_hemolysis_rows(); seqs=rbc.sequence.tolist(); labels=rbc.label.to_numpy(int)
groups=m._similarity_groups(seqs); folds=np.full(len(seqs),-1)
for fold,(_,test) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(seqs)),labels,groups)): folds[test]=fold
assert (folds>=0).all()
rbc_public=pd.DataFrame({'sequence_sha256':[h(s) for s in seqs],'label':labels,'similarity_group':groups,'held_out_fold':folds})
rbc_public.to_csv(out/'EVALUATOR_RBC_MEMBERSHIP.csv',index=False)
tp=root/'data-engineering/data/processed/toxicity.parquet'; sp=root/'data-engineering/data/processed/sequences.parquet'; inputs.extend([tp,sp,root/'data-engineering/data/processed/provenance.csv'])
tox=pd.read_parquet(tp); sq=pd.read_parquet(sp)
joined=tox.merge(sq[['sequence_id','sequence','molecular_weight_da']],on='sequence_id',validate='many_to_one')
joined['computed_label']=joined.apply(m.label_hemolysis,axis=1)
joined['sequence']=joined.sequence.str.strip().str.upper()
joined=joined[joined.sequence.isin(seqs)&joined.computed_label.notna()].copy()
joined['sequence_sha256']=joined.sequence.map(h)
joined[['sequence_sha256','assay_row_id','source_database','source_file','source_record_id']].sort_values(['sequence_sha256','assay_row_id']).to_csv(out/'EVALUATOR_RBC_ASSAY_IDS.csv',index=False)
assert set(joined.sequence_sha256)==set(rbc_public.sequence_sha256)
counts={k:len(v) for k,v in selected.items()}
assert counts=={'train':53398,'validation':810,'test':810},counts
assert len(rbc)==183
summary={'scope':'Reconstructed membership from current training script and archived local inputs; no model training or metric recomputation performed.','runtime_versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','Levenshtein']},'amp_partition_counts':counts,'amp_partitions_disjoint':True,'hemolysis_input':stats,'rbc_fold_sizes':[int(sum(folds==i)) for i in range(5)],'rbc_source_assay_rows':len(joined),'all_members_have_source_links':True,'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
recorded=json.loads((root/'shared-evaluator/reports/training_data_summary.json').read_text())
assert summary['rbc_fold_sizes']==recorded['hemolysis']['fold_sizes']
assert all(summary['input_sha256'][name]==value for name,value in recorded['input_sha256'].items())
summary['recorded_input_hashes_and_fold_sizes_match']=True
(out/'EVALUATOR_DISCLOSURE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
