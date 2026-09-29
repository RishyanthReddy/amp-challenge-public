from __future__ import annotations
import argparse, csv, hashlib, json, random, shutil, datetime
from collections import Counter, defaultdict
from pathlib import Path
from .core import *

ROOT=Path(__file__).resolve().parents[2]

def svg_bars(path, title, items):
    """Portable, dependency-free QC plot writer."""
    items=list(items); width=900; height=360; maximum=max((v for _,v in items),default=1) or 1
    bars=[]
    for i,(label,value) in enumerate(items):
        x=55+i*max(1,800//max(1,len(items))); w=max(2,760//max(1,len(items))); h=int(230*value/maximum)
        bars.append(f'<rect x="{x}" y="{290-h}" width="{w}" height="{h}" fill="#2d6a8e"/><text x="{x}" y="310" font-size="10" transform="rotate(45 {x} 310)">{label}</text>')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="20" y="28" font-size="18">{title}</text><line x1="45" y1="290" x2="850" y2="290" stroke="black"/>{"".join(bars)}</svg>',encoding='utf-8')

def source_records(raw: Path):
    rec=[]; activity=[]; toxicity=[]; inventory=[]
    def add(source,file,rid,seq,header='',meta='', ann=None):
        rec.append({'source_database':source,'source_file':str(file.relative_to(ROOT)),'source_record_id':str(rid),'original_header':header,'original_sequence':str(seq or ''),'metadata_text':meta,'is_amp':True,**(ann or {})})
    # FASTA primary sources
    for source, rel in [('AMPlify','data/raw/amplify/AMPlify_AMP_train_common.fa'),('DRAMP_antibacterial_fasta','data/raw/DRAMP/Antibacterial_amps.fasta'),('DRAMP_general_fasta','data/raw/DRAMP/general_amps.fasta'),('DBAASP_fasta','data/raw/DBAASP/peptides-fasta.txt')]:
        p=ROOT/rel; rows,enc,mal=fasta_rows(p)
        inventory.append({'source_file':rel,'file_type':'FASTA','encoding':enc,'records':len(rows),'malformed_records':mal,'columns':'id,header,sequence'})
        for r in rows:add(source,p,r['id'],r['sequence'],r['header'],r['header'])
    # CSV peptide tables
    p=raw/'dbamp/dbamp_df.csv'; rows,enc,cols=csv_rows(p); inventory.append({'source_file':str(p.relative_to(ROOT)),'file_type':'CSV','encoding':enc,'records':len(rows),'malformed_records':0,'columns':' | '.join(cols)})
    for i,r in enumerate(rows,1): add('dbAMP',p,r.get('Name') or i,r.get('Sequence',''),meta=' '.join(r.values()),ann={'is_hemolytic':r.get('Hemolytic Activity','')})
    p=raw/'DBAASP/peptides.csv'; peptides,enc,cols=csv_rows(p); inventory.append({'source_file':str(p.relative_to(ROOT)),'file_type':'CSV','encoding':enc,'records':len(peptides),'malformed_records':0,'columns':' | '.join(cols)})
    pidseq={str(r.get('ID')):r.get('SEQUENCE','') for r in peptides}
    for r in peptides:add('DBAASP_peptides',p,r.get('ID',''),r.get('SEQUENCE',''),r.get('NAME',''),' '.join(r.values()))
    # Assay tables deliberately remain observational rows
    for name, kind in [('activity-against-target-species.csv','activity'),('peptides-antibiofilm-activities.csv','activity'),('hemolytic-and-cytotoxic-activities.csv','toxicity')]:
        p=raw/'DBAASP'/name; rows,enc,cols=csv_rows(p); inventory.append({'source_file':str(p.relative_to(ROOT)),'file_type':'CSV','encoding':enc,'records':len(rows),'malformed_records':0,'columns':' | '.join(cols)})
        for ix,r in enumerate(rows,1):
            pid=str(r.get('Peptide ID','')); seq=r.get('Peptide Sequence') or pidseq.get(pid,'')
            if seq: add('DBAASP_'+kind,p,pid,seq,meta=' '.join(r.values()))
            base={'source_database':'DBAASP','source_file':str(p.relative_to(ROOT)),'source_record_id':pid,'assay_row_id':ix,'sequence_normalized':norm(seq),'raw_activity_text':json.dumps(r,ensure_ascii=False)}
            if kind=='activity':
                base.update({'target_organism':r.get('Target Species',''),'target_strain':'','assay_type':r.get('Activity Measure',''),'measurement':r.get('Activity',''),'measurement_unit':r.get('Unit',''),'mic_value_raw':r.get('Activity','') if r.get('Activity Measure','').strip().upper()=='MIC' else '', 'mic_unit_raw':r.get('Unit','') if r.get('Activity Measure','').strip().upper()=='MIC' else '', 'activity_label':r.get('Activity Measure','')}) ; activity.append(base)
            else:
                base.update({'target_cell_type':r.get('Target Cell',''),'assay_type':r.get('Activity Measure for Lysis',''),'measurement':r.get('Peptide Concentration',''),'measurement_unit':r.get('Unit',''),'toxicity_label':r.get('Activity Measure for Lysis',''),'raw_toxicity_text':base.pop('raw_activity_text')}); toxicity.append(base)
    # XLSX are separately retained so FASTA/XLSX relationships can be reported, never assumed independent
    for name in ['Antibacterial_amps.xlsx','general_amps.xlsx']:
        p=raw/'DRAMP'/name; sheets=xlsx_rows(p); n=sum(len(r) for _,r,_ in sheets); heads=' || '.join(' | '.join(h) for _,_,h in sheets)
        inventory.append({'source_file':str(p.relative_to(ROOT)),'file_type':'XLSX','encoding':'OOXML','records':n,'malformed_records':0,'columns':heads})
        for sheet, rows, heads in sheets:
            candidates=[h for h in heads if 'sequence' in h.lower() or h.lower() in ('seq','peptide')]
            ids=[h for h in heads if h.lower() in ('id','dramp id','dramp_id','accession')]
            if candidates:
                for i,r in enumerate(rows,2): add('DRAMP_'+name[:-5],p,r.get(ids[0],i) if ids else i,r.get(candidates[0],''),meta=' '.join(str(v) for v in r.values()))
    return rec,activity,toxicity,inventory

def similarity_clusters(seqs, thresholds=(0.8,0.7,0.6)):
    """Exhaustive CPU clustering for the eligible corpus using the official Indel ratio.

    The corpus is deliberately bounded to valid, reference-safe sequences.  At the
    observed size this is ~8.5m pairs, so exhaustive comparison is auditable and
    avoids an approximate candidate-generation false negative.
    """
    from rapidfuzz.fuzz import ratio
    class DSU:
        def __init__(self,n): self.p=list(range(n)); self.s=[1]*n
        def find(self,x):
            while self.p[x]!=x: self.p[x]=self.p[self.p[x]];x=self.p[x]
            return x
        def union(self,a,b):
            a,b=self.find(a),self.find(b)
            if a!=b:
                if self.s[a]<self.s[b]:a,b=b,a
                self.p[b]=a;self.s[a]+=self.s[b]
    th=tuple(sorted(thresholds,reverse=True)); uf={t:DSU(len(seqs)) for t in th}; comparisons=0
    for i,a in enumerate(seqs):
        sa=a['sequence']; la=len(sa)
        for j in range(i):
            sb=seqs[j]['sequence']; lb=len(sb)
            # Maximum possible Indel ratio occurs if the shorter sequence is a substring.
            if 2*min(la,lb)/(la+lb) < min(th): continue
            comparisons += 1; score=ratio(sa,sb)/100.0
            for t in th:
                if score>=t: uf[t].union(i,j)
    rows=[]; summary=[]; mappings={}
    for t in th:
        groups=defaultdict(list)
        for i in range(len(seqs)):groups[uf[t].find(i)].append(i)
        ordered=sorted(groups.values(),key=lambda g:min(seqs[i]['sequence_id'] for i in g))
        mapping={}
        for n,g in enumerate(ordered,1):
            cid=f'indel_{int(t*100)}_{n:06d}'
            for i in g:
                sid=seqs[i]['sequence_id'];mapping[sid]=cid
                rows.append({'sequence_id':sid,'cluster_id':cid,'threshold':t,'cluster_size':len(g),'cluster_method':'exhaustive_rapidfuzz_indel_ratio'})
        mappings[t]=mapping; summary.append({'threshold':t,'method':'exhaustive RapidFuzz Indel ratio (matches official Levenshtein.ratio semantics)','sequence_count':len(seqs),'pairs_compared':comparisons,'cluster_count':len(groups),'largest_cluster':max(map(len,groups.values()),default=0),'multi_sequence_clusters':sum(len(g)>1 for g in groups.values())})
    return rows,summary,mappings

def clustered_split(seqs, mapping, seed=42):
    groups=defaultdict(list)
    for r in seqs: groups[mapping[r['sequence_id']]].append(r)
    targets={'train':len(seqs)*.8,'validation':len(seqs)*.1,'test':len(seqs)*.1}; used={k:0 for k in targets}
    # Largest groups first, stable seeded tie-break: keeps a whole similarity component together.
    ordered=sorted(groups.items(),key=lambda x:(-len(x[1]),hashlib.sha256((str(seed)+x[0]).encode()).hexdigest()))
    out=[]
    for cid,rs in ordered:
        # Choose the assignment that minimizes total deviation from target sizes;
        # this assigns an oversized component to train rather than a small holdout.
        def deviation(candidate):
            return sum(abs((used[k] + (len(rs) if k == candidate else 0)) - targets[k]) for k in targets)
        split=min(targets,key=lambda k:(deviation(k), {'train':0,'validation':1,'test':2}[k])); used[split]+=len(rs)
        out.extend({**r,'split':split,'cluster_id':cid,'split_cluster_threshold':0.8} for r in rs)
    return out

def source_manifest(raw):
    """Local source manifest with upstream terms recorded separately from snapshot metadata."""
    catalog={
        'amplify':(
            'AMPlify training common export',
            'https://raw.githubusercontent.com/BirolLab/AMPlify/3a07713c25b8a21ef66d31d10e121989d26d9320/data/AMPlify_AMP_train_common.fa',
            '3a07713c25b8a21ef66d31d10e121989d26d9320',
            'The local FASTA is byte-identical to this pinned upstream file. The upstream '
            'GPL-3.0 program license refers to a per-file COPYRIGHT file absent at this commit; '
            'no separate dataset license was found. Confirm data rights and redistribution; '
            'do not infer them from the software license.',
        ),
        'DBAASP':(
            'DBAASP',
            'https://dbaasp.dbaasp.niaidprod.net/',
            'version not embedded in local export',
            'The current terms webpage permits use and redistribution with acknowledgement, while '
            'the separate official PDF (last modified 2026-09-24) includes a conflicting '
            'non-distribution visitor clause. The snapshot acquisition date is unknown; clarify '
            'which terms govern this export and its derivatives before release.',
        ),
        'dbamp':(
            'dbAMP',
            'https://ycclab.cuhk.edu.cn/dbAMP/download.php',
            'version not embedded in local export',
            'The provider download page links a license stating academic use is free of charge; it '
            'does not expressly grant public redistribution or address prize-bearing competition '
            'use. Snapshot version/date are unknown; confirm rights for this export and derivatives '
            'before release.',
        ),
        'DRAMP':(
            'DRAMP',
            'https://cpu-bioinfor.org/',
            'version not embedded in local export',
            'DRAMP states that its data are available under CC BY 4.0; attribute DRAMP. '
            'The pre-existing local snapshot has no recorded export date/version.',
        ),
    }
    rows=[]
    for p in sorted(raw.rglob('*')):
        if p.is_file():
            group=p.relative_to(raw).parts[0]; name,url,version,license_notes=catalog[group]
            rows.append({'source_name':name,'source_url':url,'source_version':version,'download_date':'unknown_preexisting_local_file','source_file':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'retrieval_method':'pre-existing local source export','license_notes':license_notes})
    return rows

def organism_map(activity):
    rows=[]; seen={}
    aliases={'e. coli':'Escherichia coli','escherichia coli':'Escherichia coli','s. aureus':'Staphylococcus aureus','staphylococcus aureus':'Staphylococcus aureus','p. aeruginosa':'Pseudomonas aeruginosa','pseudomonas aeruginosa':'Pseudomonas aeruginosa'}
    for r in activity:
        raw=(r.get('target_organism') or '').strip(); low=raw.lower(); canon='unknown'; strain='unknown'
        for alias,name in aliases.items():
            if low.startswith(alias):
                canon=name; strain=raw[len(alias):].strip(' ,') or 'unknown'; break
        if canon=='unknown' and raw:
            parts=raw.split(); canon=' '.join(parts[:2]) if len(parts)>=2 else raw; strain=' '.join(parts[2:]) or 'unknown'
        key=(raw,canon,strain)
        if key not in seen: seen[key]={'raw_target_organism':raw,'canonical_species':canon,'strain':strain,'mapping_method':'prefix_alias_or_first_two_tokens','mapping_status':'review_required' if canon=='unknown' else 'mapped'}
    return list(seen.values())

def main(argv=None):
    global ROOT
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); args=ap.parse_args(argv)
    ROOT=args.root.resolve(); raw=ROOT/'data/raw'; out=ROOT/'data'; proc=out/'processed'; inter=out/'intermediate'; reports=out/'reports'
    records,activity,toxicity,inventory=source_records(raw)
    # Per-record validation + provenance; sequence table remains one row per exact normalized sequence.
    canonical={}; prov=[]
    for r in records:
        q=validity(r['original_sequence'],r['metadata_text']); s=q['sequence_normalized']; sid=sequence_id(s) if s else ''
        r.update(q); r['sequence_id']=sid
        prov.append({k:r.get(k,'') for k in ['sequence_id','source_database','source_file','source_record_id','original_header','original_sequence','sequence_normalized','metadata_text']})
        if s and sid not in canonical: canonical[sid]={'sequence_id':sid,'sequence':s,'sequence_normalized':s,**q}
    refs,_,_=fasta_rows(out/'challenge/antibacterial.fasta'); refset={norm(r['sequence']) for r in refs}
    for r in canonical.values(): r['challenge_exact_overlap']=r['sequence'] in refset; r['eligible_for_generation_corpus']=r['valid_for_challenge'] and not r['challenge_exact_overlap']
    # link observation tables only after canonical IDs exist
    lookup={r['sequence_normalized']:sid for sid,r in canonical.items()}
    for table in (activity,toxicity):
        for r in table:r['sequence_id']=lookup.get(r.pop('sequence_normalized',''),'')
    for r in activity:
        if str(r.get('assay_type','')).strip().upper()=='MIC':
            r.update(parse_mic(r.get('mic_value_raw'), r.get('mic_unit_raw'), canonical.get(r['sequence_id'],{}).get('sequence','')))
        raw_note=r.get('raw_activity_text','').lower()
        if 'not active' in raw_note or 'inactive' in raw_note:
            r['activity_evidence_label']='tested_inactive'
            r['activity_evidence_basis']='source assay note explicitly reports not active under this condition'
        elif str(r.get('measurement','')).strip().upper() not in {'', 'NA'}:
            r['activity_evidence_label']='experimental_measurement_reported'
            r['activity_evidence_basis']='source assay row has a reported measurement; activity interpretation remains assay-specific'
        else:
            r['activity_evidence_label']='unknown'
            r['activity_evidence_basis']='no interpretable activity conclusion in source assay row'
    # Preserve original target text and expose a conservative species/strain mapping.
    org_rows=organism_map(activity); org_lookup={r['raw_target_organism']:(r['canonical_species'],r['strain']) for r in org_rows}
    for r in activity:
        r['canonical_species'],r['canonical_strain']=org_lookup.get(r.get('target_organism',''),('unknown','unknown'))
    annotations=[]
    bysid=defaultdict(list)
    for r in records:
        if r['sequence_id']: bysid[r['sequence_id']].append(r)
    for sid,rs in bysid.items():
        text=' '.join(r['metadata_text'].lower() for r in rs)
        annotations.append({'sequence_id':sid,'is_amp':True,'is_antibacterial':any('antibacterial' in r['source_database'].lower() for r in rs),'is_antibiofilm':'antibiofilm' in text,'is_hemolytic':'hemol' in text,'is_cytotoxic':'cytotox' in text,'target_gram_status':''})
    seqs=list(canonical.values())
    for r in seqs:
        r.update(biophysical_descriptors(r['sequence']))
    # The official reference constrains final generated candidates, not model training.
    core_training=[r for r in seqs if r['valid_for_challenge']]
    eligible=[r for r in seqs if r['eligible_for_generation_corpus']]
    cl, cluster_summary, mappings=similarity_clusters(eligible)
    split=clustered_split(eligible,mappings[0.8]); split_by={x:[r for r in split if r['split']==x] for x in ['train','validation','test']}
    holdout_all_ids={r['sequence_id'] for r in split_by['validation']} | {r['sequence_id'] for r in split_by['test']}
    train_rows=[r for r in core_training if r['sequence_id'] not in holdout_all_ids]
    for name, rows in [('sequences',seqs),('provenance',prov),('activity',activity),('toxicity',toxicity),('annotations',annotations),('sequence_clusters',cl),('train',train_rows),('validation',split_by['validation']),('test',split_by['test']),('novelty_safe_train',split_by['train']),('novelty_safe_validation',split_by['validation']),('novelty_safe_test',split_by['test'])]:
        write_csv(proc/(name+'.csv'),rows); write_parquet(proc/(name+'.parquet'),rows)
    (proc/'unique_amp_sequences.fasta').parent.mkdir(parents=True,exist_ok=True)
    with (proc/'unique_amp_sequences.fasta').open('w',encoding='utf-8') as f:
        for r in eligible:f.write(f">{r['sequence_id']}\n{r['sequence']}\n")
    # task-specific views are references/copies so downstream task boundaries are explicit
    for folder in ['generation','classification','activity_prediction','ranking']:(proc/folder).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(proc/'unique_amp_sequences.fasta',proc/'generation/amp_sequences.fasta')
    # Positive-only until a reviewed, versioned negative-control source is approved.
    classification_train=[{**r,'is_amp':True,'label_scope':'positive_only_no_negative_controls'} for r in train_rows]
    write_csv(proc/'classification/train.csv',classification_train); write_parquet(proc/'classification/train.parquet',classification_train)
    for nme in ['validation','test']:
        rows=[{**r,'is_amp':True,'label_scope':'positive_only_no_negative_controls'} for r in split_by[nme]]
        write_csv(proc/f'classification/{nme}.csv',rows); write_parquet(proc/f'classification/{nme}.parquet',rows)
    core_ids={r['sequence_id'] for r in core_training}; holdout_ids={x:{r['sequence_id'] for r in split_by[x]} for x in ['validation','test']}; seq_fields={r['sequence_id']:r for r in seqs}
    def joined_activity(ids):
        return [{**r,'sequence':seq_fields[r['sequence_id']]['sequence'],'length':seq_fields[r['sequence_id']]['length'],'alphabet_valid':seq_fields[r['sequence_id']]['alphabet_valid'],'is_regression_ready':r.get('mic_value_uM') is not None} for r in activity if r.get('sequence_id') in ids and str(r.get('assay_type','')).strip().upper()=='MIC']
    activity_train=joined_activity(core_ids-holdout_ids['validation']-holdout_ids['test'])
    for nme,rows in [('train',activity_train),('validation',joined_activity(holdout_ids['validation'])),('test',joined_activity(holdout_ids['test']))]:
        write_csv(proc/f'activity_prediction/{nme}.csv',rows); write_parquet(proc/f'activity_prediction/{nme}.parquet',rows)
    all_activity=joined_activity(core_ids); write_csv(proc/'activity_prediction/all_observations.csv',all_activity); write_parquet(proc/'activity_prediction/all_observations.parquet',all_activity)

    # Activity-specific split: only labelled MIC sequence IDs, clustered independently,
    # so validation/test contain observations without modifying core or novelty-safe views.
    mic_ids=sorted({r['sequence_id'] for r in all_activity if r.get('sequence_id')})
    mic_seqs=[seq_fields[s] for s in mic_ids]
    mic_cl, mic_summary, mic_maps=similarity_clusters(mic_seqs)
    mic_split=clustered_split(mic_seqs,mic_maps[.8]); mic_by={x:{r['sequence_id'] for r in mic_split if r['split']==x} for x in ['train','validation','test']}
    activity_views=[]
    for nme in ['train','validation','test']:
        rows=joined_activity(mic_by[nme]); activity_views.extend({**r,'activity_split':nme,'activity_split_cluster_id':mic_maps[.8][r['sequence_id']]} for r in rows)
        outrows=[{**r,'activity_split':nme,'activity_split_cluster_id':mic_maps[.8][r['sequence_id']]} for r in rows]
        write_csv(proc/f'activity_prediction/{nme}.csv',outrows); write_parquet(proc/f'activity_prediction/{nme}.parquet',outrows)
    write_csv(reports/'activity_split_statistics.csv',[{'split':x,'sequence_count':len(mic_by[x]),'observation_count':sum(1 for r in activity_views if r['activity_split']==x),'cluster_threshold':.8,'cluster_method':'exhaustive_rapidfuzz_indel_ratio'} for x in ['train','validation','test']])
    write_csv(reports/'activity_sequence_clusters.csv',mic_cl)
    def leakage_row(name, groups, mapping):
        owners=defaultdict(set)
        for split_name, ids in groups.items():
            for sid in ids: owners[mapping[sid]].add(split_name)
        return {'split_policy':name,'threshold':.8,'sequence_count':sum(len(v) for v in groups.values()),'cluster_count':len(owners),'cross_split_cluster_count':sum(len(v)>1 for v in owners.values()),'status':'pass' if not any(len(v)>1 for v in owners.values()) else 'fail'}
    write_csv(reports/'leakage_check.csv',[leakage_row('novelty_safe', {x:{r['sequence_id'] for r in split_by[x]} for x in split_by}, mappings[.8]),leakage_row('activity_mic',mic_by,mic_maps[.8])])
    write_csv(reports/'dataset_splits.csv',[{'dataset_view':'novelty_safe','sequence_id':r['sequence_id'],'split':r['split'],'cluster_id':r['cluster_id'],'threshold':.8} for r in split]+[{'dataset_view':'activity_mic','sequence_id':r['sequence_id'],'split':r['split'],'cluster_id':mic_maps[.8][r['sequence_id']],'threshold':.8} for r in mic_split])

    # Role-specific model views generated from master tables, not manually maintained copies.
    role=proc/'views'; role.mkdir(parents=True,exist_ok=True)
    split_lookup={r['sequence_id']:r for r in split}
    descriptor_names=['net_charge_ph7','isoelectric_point','molecular_weight_da','eisenberg_hydrophobic_moment','boman_index','grand_avg_hydropathy','instability_index']
    def role_base(r):
        x=split_lookup.get(r['sequence_id']); return {'sequence_id':r['sequence_id'],'sequence':r['sequence'],'length':r['length'],'split':x['split'] if x else 'core_train_only','cluster_id':x['cluster_id'] if x else 'not_clustered_reference_overlap_or_non_novelty_safe','data_version':'1.0'}
    def short_descriptors(r): return {k:r[k] for k in ['net_charge_ph7','eisenberg_hydrophobic_moment','boman_index']}
    def all_descriptors(r): return {k:r[k] for k in descriptor_names}
    ar=[{**role_base(r),**short_descriptors(r)} for r in core_training]
    write_csv(role/'autoregressive_view.csv',ar); write_parquet(role/'autoregressive_view.parquet',ar)
    mic_by_seq=defaultdict(list)
    for r in all_activity: mic_by_seq[r['sequence_id']].append(r)
    vae=[{**role_base(r),**short_descriptors(r),'is_amp':True,'mic_label_uM':min((x['mic_value_uM'] for x in mic_by_seq.get(r['sequence_id'],[]) if x.get('mic_value_uM') is not None),default=None),'hydramp_length_eligible':r['length']<=25} for r in core_training]
    write_csv(role/'vae_latent_view.csv',vae); write_parquet(role/'vae_latent_view.parquet',vae)
    diffusion=[]
    for r in core_training:
        best_obs=min((x for x in mic_by_seq.get(r['sequence_id'], []) if x.get('mic_value_uM') is not None), key=lambda x:x['mic_value_uM'], default=None)
        diffusion.append({**role_base(r), **short_descriptors(r),
                          'optional_condition_mic_uM':best_obs['mic_value_uM'] if best_obs else None,
                          'optional_condition_species':best_obs['canonical_species'] if best_obs and best_obs.get('canonical_species') != 'unknown' else 'unknown'})
    write_csv(role/'diffusion_view.csv',diffusion); write_parquet(role/'diffusion_view.parquet',diffusion)
    # One deterministic reference-safe representative per 80% cluster; natural status is not claimed without a dedicated source field.
    bycluster=defaultdict(list)
    for r in eligible: bycluster[mappings[.8][r['sequence_id']]].append(r)
    evo=[]
    for cid,members in sorted(bycluster.items()):
        r=min(members,key=lambda x:hashlib.sha256(x['sequence_id'].encode()).hexdigest()); psrc=sorted({p['source_database'] for p in prov if p['sequence_id']==r['sequence_id']})
        evo.append({**role_base(r),**all_descriptors(r),'source_databases':';'.join(psrc),'activity_observation_count':len(mic_by_seq.get(r['sequence_id'],[])),'toxicity_observation_count':sum(x.get('sequence_id')==r['sequence_id'] for x in toxicity),'official_reference_exact_overlap':r['challenge_exact_overlap'],'inclusion_reason':'one_deterministic_representative_per_80pct_cluster; diverse_template; natural_status_not_asserted'})
    write_csv(role/'evolution_seed_view.csv',evo); write_parquet(role/'evolution_seed_view.parquet',evo)
    # Functional motifs and curated seeds draw from the entire validated training
    # corpus with MIC observations, not only the novelty-safe cluster representatives.
    full_seed_rows=[]
    for r in core_training:
        n_mic=len(mic_by_seq.get(r['sequence_id'],[]))
        if not n_mic: continue
        psrc=sorted({p['source_database'] for p in prov if p['sequence_id']==r['sequence_id']})
        full_seed_rows.append({**role_base(r),**all_descriptors(r),'source_databases':';'.join(psrc),'activity_observation_count':n_mic,'toxicity_observation_count':sum(x.get('sequence_id')==r['sequence_id'] for x in toxicity),'official_reference_exact_overlap':r['challenge_exact_overlap'],'inclusion_reason':'validated_training_sequence_with_local_MIC_observation; natural_status_not_asserted'})
    # Functional motifs: count each k-mer occurrence across all activity-supported training sequences.
    motif_occ=Counter(); motif_ids=defaultdict(set); motif_charges=defaultdict(list)
    for r in full_seed_rows:
        seq=r['sequence']
        for k in range(4,9):
            for i in range(len(seq)-k+1):
                motif=seq[i:i+k]
                cationic=[j for j,a in enumerate(motif) if a in 'KR']; hydrophobic=[j for j,a in enumerate(motif) if a in 'WFLIV']
                if not any(abs(a-b)<=3 for a in cationic for b in hydrophobic): continue
                motif_occ[motif]+=1; motif_ids[motif].add(r['sequence_id']); motif_charges[motif].append(net_charge_ph7(motif))
    motifs=[]
    for motif,count in motif_occ.items():
        if len(motif_ids[motif])>=3:
            motifs.append({'motif_sequence':motif,'frequency':count,'mean_charge':round(statistics.mean(motif_charges[motif]),6),'associated_seq_count':len(motif_ids[motif]),'example_sequence_ids':';'.join(sorted(motif_ids[motif])[:5])})
    motifs.sort(key=lambda r:(-r['associated_seq_count'],-r['frequency'],r['motif_sequence']))
    write_csv(proc/'motif_library.csv',motifs,fields=['motif_sequence','frequency','mean_charge','associated_seq_count','example_sequence_ids'])
    # Curated panel: filter, then deterministic max-min farthest point sampling over z-scored descriptor space.
    candidates=[r for r in full_seed_rows if 2<=r['net_charge_ph7']<=8 and r['eisenberg_hydrophobic_moment']>=.2]
    dims=['net_charge_ph7','eisenberg_hydrophobic_moment','length']
    means={d:statistics.mean(r[d] for r in candidates) for d in dims}; stds={d:statistics.pstdev(r[d] for r in candidates) or 1 for d in dims}
    def distance(a,b): return sum(((a[d]-b[d])/stds[d])**2 for d in dims)**.5
    chosen=[]
    if candidates:
        chosen.append(min(candidates,key=lambda r:r['sequence_id']))
    remaining=[r for r in candidates if r not in chosen]
    while remaining and len(chosen)<100:
        pick=max(remaining,key=lambda r:(min(distance(r,x) for x in chosen),r['sequence_id']))
        chosen.append(pick); remaining.remove(pick)
    def track(r):
        if r['length']<=25 and r['eisenberg_hydrophobic_moment']>=.4:return 'all'
        if r['length']<=25:return 'vae'
        if r['net_charge_ph7']>=5:return 'ar'
        if r['eisenberg_hydrophobic_moment']>=.45:return 'diffusion'
        return 'evolution'
    panel=[{**r,'recommended_generator_track':track(r),'seed_selection_rank':i,'selection_method':'filtered_80pct_cluster_representatives_then_maxmin_farthest_point_sampling'} for i,r in enumerate(chosen,1)]
    write_csv(proc/'curated_seed_panel.csv',panel)
    (reports/'seed_panel_report.md').write_text(f'''# Curated evolution seed panel\n\nThe panel contains **{len(panel)}** deterministic seeds selected from {len(full_seed_rows):,} validated training sequences with at least one local MIC observation. {len(candidates):,} sequences satisfy net charge pH 7.4 from +2 to +8 and Eisenberg hydrophobic moment >=0.2.\n\nFinal selection uses deterministic max-min farthest-point sampling in z-scored (charge, hydrophobic moment, length) space, yielding a 100-member spread rather than selecting only the lowest-MIC peptides.\n\n`motif_library.csv` has {len(motifs):,} qualifying 4-8 residue motifs, each observed in at least three activity-supported sequences and containing a K/R plus W/F/L/I/V within four positions.\n\nNatural origin is not asserted: available local provenance is not a reliable natural/synthetic classifier. The panel is activity-supported and diverse, not verified natural-only.\n''',encoding='utf-8')
    toxicity_by=defaultdict(list)
    for r in toxicity: toxicity_by[r.get('sequence_id')].append(r)
    evaluator=[]
    for r in core_training:
        ms=mic_by_seq.get(r['sequence_id'],[]); ts=toxicity_by.get(r['sequence_id'],[])
        evaluator.append({**role_base(r),**all_descriptors(r),'is_amp':True,'negative_label':'unknown_no_global_negative_control_source','tested_inactive_observation_count':sum(x.get('activity_evidence_label')=='tested_inactive' for x in ms),'mic_observation_count':len(ms),'best_exact_mic_uM':min((x['mic_value_uM'] for x in ms if x.get('mic_value_uM') is not None),default=None),'organisms':';'.join(sorted({x['canonical_species'] for x in ms if x.get('canonical_species')})),'strains':';'.join(sorted({x['canonical_strain'] for x in ms if x.get('canonical_strain')})),'toxicity_observation_count':len(ts),'missing_mic':not bool(ms),'missing_toxicity':not bool(ts),'challenge_exact_overlap':r['challenge_exact_overlap']})
    write_csv(role/'evaluator_view.csv',evaluator); write_parquet(role/'evaluator_view.parquet',evaluator)
    shutil.copyfile(proc/'sequences.parquet',proc/'ranking/candidates.parquet')
    shutil.copyfile(proc/'sequences.csv',proc/'ranking/candidates.csv')
    # reports
    write_csv(reports/'dataset_inventory.csv',inventory); write_csv(reports/'raw_dataset_inventory.csv',inventory)
    source_manifest_path=reports/'source_manifest.csv'
    write_csv(
        source_manifest_path,
        source_manifest(raw),
        fields=[
            'source_name','source_url','source_version','download_date','source_file',
            'bytes','sha256','retrieval_method','license_notes',
        ],
    )
    # Keep this checked-in provenance report on LF line endings on every platform.
    with source_manifest_path.open('r',encoding='utf-8',newline='') as f:
        source_manifest_text=f.read().replace('\r\n','\n')
    with source_manifest_path.open('w',encoding='utf-8',newline='') as f:
        f.write(source_manifest_text)
    write_csv(reports/'organism_mapping.csv',org_rows)
    write_csv(reports/'official_reference_matches.csv',[{'sequence_id':r['sequence_id'],'sequence':r['sequence'],'challenge_reference':'data/challenge/antibacterial.fasta','match_type':'exact'} for r in seqs if r['challenge_exact_overlap']])
    norm_rows=[]
    categories=[('valid_core_training',lambda r:r['valid_for_challenge'],lambda r:r['valid_for_challenge']),('corrected_formatting',lambda r:r['sequence_normalized']!=str(r['original_sequence']).replace(' ','').upper(),lambda r:False),('modified',lambda r:r['modification_status']=='modified',lambda r:r['modification_status']=='modified'),('non_standard',lambda r:not r['alphabet_valid'],lambda r:not r['alphabet_valid']),('out_of_range',lambda r:not r['length_valid'],lambda r:not r['length_valid']),('unreadable_empty',lambda r:not r['sequence_normalized'],lambda r:not r['sequence_normalized']),('lowercase_suspected_d_or_mixed',lambda r:r.get('has_lowercase_residues',False),lambda r:r.get('has_lowercase_residues',False))]
    for label,record_pred,sequence_pred in categories: norm_rows.append({'category':label,'record_count':sum(record_pred(r) for r in records),'canonical_sequence_count':sum(sequence_pred(r) for r in seqs),'definition':'record-level and canonical counts can differ because provenance is retained; canonical corrected_formatting is not applicable'})
    write_csv(reports/'normalization_report.csv',norm_rows)
    (reports/'unit_conversion_rules.md').write_text('# MIC unit conversion rules\n\nRaw MIC text and units are always retained. Numeric parsing records `=`, `<`, `>`, or `range`; no exact value is invented for censored/range measurements. `µM` is retained unchanged. `µg/mL` and `mg/L` are converted using `µM = (mass concentration in µg/mL × 1000) / molecular_weight_Da`; `mg/mL` uses ×1,000,000. Molecular weight is the sum of standard residue masses plus 18.0153 Da for free termini. Conversion is emitted only with a recognized unit, parsable value, and valid canonical sequence.\n',encoding='utf-8')
    qc=[{'sequence_id':r['sequence_id'],'length':r['length'],'alphabet_valid':r['alphabet_valid'],'length_valid':r['length_valid'],'invalid_residues':r['invalid_residues'],'modification_status':r['modification_status'],'challenge_exact_overlap':r['challenge_exact_overlap'],'eligible_for_generation_corpus':r['eligible_for_generation_corpus']} for r in seqs]; write_csv(reports/'sequence_qc.csv',qc)
    srcsets=defaultdict(set)
    for r in records:
        if r['sequence_id']:srcsets[r['source_database']].add(r['sequence_id'])
    overlap=[]
    for source_a,sa in srcsets.items():
        for source_b,sb in srcsets.items():overlap.append({'source_a':source_a,'source_b':source_b,'overlap_sequences':len(sa&sb),'jaccard':round(len(sa&sb)/len(sa|sb),6) if sa|sb else 0})
    write_csv(reports/'overlap_matrix.csv',overlap)
    missing=[]
    for table,rows in [('activity',activity),('toxicity',toxicity),('provenance',prov)]:
        for k in sorted({k for r in rows for k in r}):missing.append({'table':table,'column':k,'missing_count':sum(not str(r.get(k,'')).strip() for r in rows),'records':len(rows)})
    write_csv(reports/'missingness_report.csv',missing)
    write_csv(reports/'cluster_statistics.csv',cluster_summary)
    write_csv(reports/'split_statistics.csv',[{'view':'core_training','split':'train','sequences':len(core_training),'challenge_reference_filtered':False,'seed':42,'cluster_method':'not_applicable'}]+[{'view':'novelty_safe','split':x,'sequences':len(split_by[x]),'challenge_reference_filtered':True,'seed':42,'cluster_method':'exhaustive_rapidfuzz_indel_ratio','threshold':0.8} for x in ['train','validation','test']])
    descriptor_rows=[]
    for field in ['net_charge_ph7','isoelectric_point','molecular_weight_da','eisenberg_hydrophobic_moment','boman_index','grand_avg_hydropathy','instability_index']:
        vals=[r[field] for r in core_training if r.get(field) is not None]
        descriptor_rows.append({'descriptor':field,'sequence_count':len(vals),'min':min(vals),'max':max(vals),'mean':round(statistics.mean(vals),6),'std':round(statistics.pstdev(vals),6)})
    write_csv(reports/'biophysical_descriptor_report.csv',descriptor_rows)
    counts=Counter(r['source_database'] for r in records); valid=sum(r['valid_for_challenge'] for r in seqs); eligible_n=len(eligible)
    lengths=Counter(r['length'] for r in seqs); amino=Counter(''.join(r['sequence'] for r in eligible)); acts=Counter(r.get('activity_label','') or 'missing' for r in activity)
    svg_bars(reports/'figures/sequence_length_distribution.svg','Canonical sequence length distribution',sorted(lengths.items()))
    svg_bars(reports/'figures/amino_acid_composition.svg','Generation-corpus amino-acid composition',sorted(amino.items()))
    svg_bars(reports/'figures/source_dataset_counts.svg','Standardized source record counts',sorted(counts.items()))
    svg_bars(reports/'figures/activity_label_distribution.svg','DBAASP activity observation labels',acts.most_common(25))
    c80=Counter(r['cluster_size'] for r in cl if r['threshold']==.8)
    svg_bars(reports/'figures/cluster_size_distribution.svg','80% identity cluster-size distribution',sorted((str(k),v) for k,v in c80.items()))
    svg_bars(reports/'figures/overlap_between_databases.svg','Within-source canonical sequence counts',sorted((k,len(v)) for k,v in srcsets.items()))
    reports.mkdir(parents=True,exist_ok=True)
    (reports/'raw_dataset_inspection.md').write_text('# Raw dataset inspection\n\nGenerated programmatically. `dataset_inventory.csv` records encoding, type, row/record count and headers for every immutable raw source. XLSX values are read from OOXML sheets without modifying workbooks.\n\n'+ '\n'.join(f'- {k}: {v} standardized source records' for k,v in sorted(counts.items())),encoding='utf-8')
    (reports/'data_quality_report.md').write_text(f'''# Data quality report\n\n- Source-standardized records: {len(records):,}\n- Unique canonical normalized sequences: {len(seqs):,}\n- Valid 8–50 aa, standard-alphabet, no-known-modification sequences: {valid:,}\n- Eligible generation corpus (also excludes exact challenge-reference matches): {eligible_n:,}\n- Challenge reference sequences: {len(refset):,}; exact overlap flagged: {sum(r['challenge_exact_overlap'] for r in seqs):,}\n- Source-record-to-canonical surplus (repeated assays and sequence duplicates): {len(records)-len(seqs):,}\n- Activity observations: {len(activity):,}; toxicity observations: {len(toxicity):,}\n\n## Similarity and leakage\n\nThe challenge-safe generation corpus is exhaustively compared pairwise using RapidFuzz's Indel similarity, which matches the official validator's `Levenshtein.ratio` semantics. Connected components are written for 80%, 70%, and 60% thresholds. Train/validation/test assignment is deterministic (seed 42) and keeps every 80%-similarity component in a single split.\n\n## APD3 and AMPlify\n\nAPD3 is not a standalone local source. AMPlify is preserved as a consolidated source and quantified in `overlap_matrix.csv`; it is not counted as independent biological evidence.\n''',encoding='utf-8')
    print(json.dumps({'canonical_sequences':len(seqs),'core_training_sequences':len(core_training),'eligible_sequences':eligible_n,'novelty_safe_splits':{x:len(split_by[x]) for x in split_by},'cluster_statistics':cluster_summary},indent=2))
if __name__=='__main__':main()
