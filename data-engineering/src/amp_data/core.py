from __future__ import annotations

import base64, csv, hashlib, io, json, math, re, statistics, zlib
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
MOD_WORDS = ("amidat", "acetyl", "lipid", "glycosyl", "peg", "cycl", "stapl", "dendr", "noncanonical", "peptidomimetic", "terminal mod", "disulfide")

def read_text(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try: return data.decode(enc), enc
        except UnicodeDecodeError: pass
    return data.decode("latin-1", errors="replace"), "latin-1-replace"

def csv_rows(path: Path):
    text, enc = read_text(path)
    rows = list(csv.DictReader(io.StringIO(text, newline='')))
    return rows, enc, list(rows[0]) if rows else []

def fasta_rows(path: Path):
    text, enc = read_text(path); out=[]; head=None; seq=[]; malformed=0
    for line in text.splitlines():
        if line.startswith(">"):
            if head is not None: out.append({"id":head.split()[0].strip(), "header":head, "sequence":''.join(seq)})
            head=line[1:].strip(); seq=[]
        elif line.strip():
            if head is None: malformed += 1
            else: seq.append(line.strip())
    if head is not None: out.append({"id":head.split()[0].strip(), "header":head, "sequence":''.join(seq)})
    return out, enc, malformed

def xlsx_rows(path: Path):
    """Small dependency-free XLSX reader for values-only inspection/standardization."""
    ns='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared=[''.join(x.itertext()) for x in root.findall(ns+'si')]
        sheets=[]
        for n in sorted(x for x in z.namelist() if x.startswith('xl/worksheets/sheet') and x.endswith('.xml')):
            root=ET.fromstring(z.read(n)); records=[]
            for row in root.findall('.//'+ns+'row'):
                vals=[]
                for c in row.findall(ns+'c'):
                    v=c.find(ns+'v'); value='' if v is None else v.text or ''
                    if c.attrib.get('t')=='s' and value.isdigit(): value=shared[int(value)]
                    elif c.attrib.get('t')=='inlineStr': value=''.join(c.itertext())
                    ref=c.attrib.get('r','')
                    match=re.match(r'([A-Z]+)', ref)
                    if match:
                        col=0
                        for letter in match.group(1): col=col*26 + ord(letter)-ord('A')+1
                        col-=1
                        if len(vals)<=col: vals.extend(['']*(col+1-len(vals)))
                        vals[col]=value
                    else:
                        vals.append(value)
                records.append(vals)
            if records:
                h=[str(v).strip() or f'column_{i+1}' for i,v in enumerate(records[0])]
                sheets.append((n, [dict(zip(h, r+['']*(len(h)-len(r)))) for r in records[1:]], h))
    return sheets

def norm(seq: str) -> str:
    return re.sub(r'\s+', '', str(seq or '')).upper()

def validity(seq: str, metadata: str='') -> dict:
    raw=str(seq or ''); lower=any(c.islower() for c in raw)
    s=norm(raw); invalid=''.join(sorted(set(s)-ALPHABET))
    meta=metadata.lower(); mods=[w for w in MOD_WORDS if w in meta]
    # DBAASP exports use AMD in terminus fields for amidation.
    if re.search(r'\bamd\b', meta): mods.append('amidation (AMD)')
    status='modified' if mods else 'unknown' if not metadata.strip() else 'none_known'
    return {'sequence_normalized':s, 'length':len(s), 'alphabet_valid':not bool(invalid),
            'has_lowercase_residues':lower,
            'stereochemistry_status':'suspected_d_or_mixed' if lower else 'l_or_unspecified',
            'invalid_residues':invalid, 'length_valid':8<=len(s)<=50,
            'modification_status':status, 'modification_reason':';'.join(mods),
            'valid_for_challenge': bool(s) and not invalid and 8<=len(s)<=50 and not mods and not lower}

RESIDUE_MASS = {'A':71.0788,'C':103.1388,'D':115.0886,'E':129.1155,'F':147.1766,'G':57.0519,'H':137.1411,'I':113.1594,'K':128.1741,'L':113.1594,'M':131.1926,'N':114.1038,'P':97.1167,'Q':128.1307,'R':156.1875,'S':87.0782,'T':101.1051,'V':99.1326,'W':186.2132,'Y':163.1760}
EISENBERG = {'A':.62,'R':-2.53,'N':-.78,'D':-.90,'C':.29,'Q':-.85,'E':-.74,'G':.48,'H':-.40,'I':1.38,'L':1.06,'K':-1.50,'M':.64,'F':1.19,'P':.12,'S':-.18,'T':-.05,'W':.81,'Y':.26,'V':1.08}
KYTE_DOOLITTLE = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-.8,'T':-.7,'W':-.9,'Y':-1.3,'V':4.2}
# Radzicka-Wolfenden transfer free energies (cyclohexane -> water), used by
# the conventional Boman protein-binding-potential index.
BOMAN = {'L':-4.92,'I':-4.92,'V':-4.04,'F':-2.98,'M':-2.35,
         'W':-2.33,'A':-1.81,'C':-1.28,'G':-.94,'Y':-.14,
         'T':2.57,'S':3.40,'H':4.66,'Q':5.54,'K':5.55,
         'N':6.64,'E':6.81,'D':8.72,'R':9.38,'P':0.00}
_DIWV_B64='eNq1l91uwjAMhd8l1xBRWqBwN439CZhgICaeZeq7b7VPErd10ha0G1ylSfr19NgxP+bJbH7qn8zOJubZbIrCrouJ2ZrNdGWL9cS84N4r4hviu5/xgZEd4h7xgPiJeDSb+czOlxNzwsgX4hnxgnhF/Ea8UawIsEFLcev3TbDmuV2qqFh6cFNU2unSLgqVlxdd/ZQ/5Hlhl6WA3kagKQpk7CCgKQpmCJ4QmKKQF3ue/cv8IU+zws7yhMovUU8Enfm9NVPwEz/85FG2wJUQOnC3reHew3O/xoTObV4k3dERGlv/i5Vr6WZZRQDEi8+qWANiSmJ+F8GM1Ul74I5ikDY0ZkpsfuIN9yp6+DBDr22+ImpcCWp4aufTRaHGnWP9PWxZprlZqYY9aM0Nz6rosUluMGn+YA3UVAwFpG2RfmpnEezWycTdQKUdL/ZR3eGSECNawYOIJ5aeCx5PGsG8H8l8f7ELKciQX34okYXt6nxwuPyJRwAvSjsvk8cfDCCQYbHGkRJqNO5e/EqlehB/RZs2ZMYKLQ1RzN7C5chE7Bo6fiB2CwiS3yt+BDm+1bPfbeuvavrS5iXxY17CKh3deZeeE0Y7G51bMNIoI0x/auvu2cPhiKtBZ/ojZ2Pwi09K34U4v9BIRcvuLCV47hBmzqEINfJF5XZewRzBfW5xc6oL8CB2X0XpkzokZ4AOBSXeiDTr36VHaMHLcql1u4/ZZbI40uM52ds5XSPMbuKgoyY0T5QxD/VOShvS8vO3a56AOMTSvgtBqdN05m9y8HMatu7tnkJrHTseb6768fbyTwyPqG2fULvTirQtAhvLfwW8Rv4vyBZ2nQ0v3PUGVfULJOzGnA=='
GURUPRASAD_DIWV=json.loads(zlib.decompress(base64.b64decode(_DIWV_B64)).decode())

def molecular_weight_da(sequence: str):
    if not sequence or set(sequence)-set(RESIDUE_MASS): return None
    return round(sum(RESIDUE_MASS[x] for x in sequence)+18.0153, 4)

def net_charge_ph7(sequence: str, ph: float=7.4):
    if not sequence or set(sequence)-ALPHABET:return None
    pos=1/(1+10**(ph-9.69))+sum(sequence.count(a)/(1+10**(ph-p)) for a,p in {'K':10.5,'R':12.4,'H':6.0}.items())
    neg=1/(1+10**(2.34-ph))+sum(sequence.count(a)/(1+10**(p-ph)) for a,p in {'D':3.86,'E':4.25,'C':8.33,'Y':10.07}.items())
    return round(pos-neg,4)
def isoelectric_point(sequence: str):
    if not sequence or set(sequence)-ALPHABET:return None
    lo,hi=0.,14.
    for _ in range(50):
        mid=(lo+hi)/2
        if net_charge_ph7(sequence,mid)>0:lo=mid
        else:hi=mid
    return round((lo+hi)/2,4)
def eisenberg_hydrophobic_moment(sequence: str,window:int=11):
    if not sequence or set(sequence)-ALPHABET:return None
    parts=[sequence] if len(sequence)<=window else [sequence[i:i+window] for i in range(len(sequence)-window+1)]
    return round(max(math.hypot(sum(EISENBERG[a]*math.cos(math.radians(i*100)) for i,a in enumerate(x)),sum(EISENBERG[a]*math.sin(math.radians(i*100)) for i,a in enumerate(x)))/len(x) for x in parts),6)
def boman_index(sequence: str): return round(sum(BOMAN[a] for a in sequence)/len(sequence),2) if sequence and not(set(sequence)-ALPHABET) else None
def gravy(sequence: str): return round(sum(KYTE_DOOLITTLE[a] for a in sequence)/len(sequence),6) if sequence and not(set(sequence)-ALPHABET) else None
def instability_index(sequence: str): return round(10*sum(GURUPRASAD_DIWV[a][b] for a,b in zip(sequence,sequence[1:]))/len(sequence),6) if sequence and not(set(sequence)-ALPHABET) else None
def biophysical_descriptors(sequence: str): return {'net_charge_ph7':net_charge_ph7(sequence),'isoelectric_point':isoelectric_point(sequence),'molecular_weight_da':molecular_weight_da(sequence),'eisenberg_hydrophobic_moment':eisenberg_hydrophobic_moment(sequence),'boman_index':boman_index(sequence),'grand_avg_hydropathy':gravy(sequence),'instability_index':instability_index(sequence)}

def parse_mic(value, unit, sequence):
    """Parse DBAASP MIC without inventing a single value for censored/range observations."""
    raw=str(value or '').strip(); u=str(unit or '').strip().lower().replace('μ','u').replace('µ','u').replace('�','u').replace(' ','')
    # A hyphen between numeric endpoints (with optional surrounding whitespace)
    # is a range delimiter, not the sign of its endpoint.
    numeric_text=re.sub(r'(?<=\d)\s*-\s*(?=\d)', ' ', raw)
    nums=[float(x) for x in re.findall(r'(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', numeric_text)]
    qualifier='<' if raw.startswith('<') else '>' if raw.startswith('>') else 'range' if len(nums)>=2 else '=' if nums else ''
    if qualifier=='<':
        lower=None; upper=nums[0] if nums else None
    elif qualifier=='>':
        lower=nums[0] if nums else None; upper=None
    elif qualifier=='range':
        lower=min(nums); upper=max(nums)
    else:
        lower=upper=nums[0] if nums else None
    mw=molecular_weight_da(sequence); factor=None
    if u in {'um','μm'}: factor=1.0
    elif u in {'ug/ml','ug/ml.'}: factor=1000/mw if mw else None
    elif u in {'mg/l','mg/l.'}: factor=1000/mw if mw else None
    elif u in {'mg/ml'}: factor=1_000_000/mw if mw else None
    status='converted' if factor is not None else 'unsupported_or_missing_unit' if nums else 'unparseable_value'
    return {'mic_value_numeric':lower if qualifier!='range' else None,'mic_lower_bound':lower,'mic_upper_bound':upper,
            'mic_qualifier':qualifier,'mic_unit_normalized':u,'molecular_weight_da':mw,
            'mic_value_uM':round(lower*factor,6) if factor is not None and qualifier=='=' else None,
            'mic_lower_bound_uM':round(lower*factor,6) if factor is not None and lower is not None else None,
            'mic_upper_bound_uM':round(upper*factor,6) if factor is not None and upper is not None else None,
            'mic_conversion_status':status}

def sequence_id(s: str) -> str: return 'seq_'+hashlib.sha256(s.encode()).hexdigest()[:16]
def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True); rows=list(rows)
    if fields is None: fields=sorted({k for r in rows for k in r})
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def write_parquet(path: Path, rows):
    """Write the requested model-ready Parquet artifact (PyArrow is project-managed)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows=list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    normalized=[{k:r.get(k) for k in fields} for r in rows]
    pq.write_table(pa.Table.from_pylist(normalized), path, compression='zstd')

def lev_ratio(a: str,b: str) -> float:
    """Official AMP validator metric: normalized Indel similarity, not edit/max-length."""
    from rapidfuzz.fuzz import ratio
    return ratio(a, b) / 100.0

def challenge_matches(sequences, refs, threshold=0.8):
    out=[]
    for s in sequences:
        best=max((lev_ratio(s,r) for r in refs), default=0.0)
        if best>threshold: out.append((s,best))
    return out
