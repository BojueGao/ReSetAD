"""Versioned RR-Rank workspace; immutable parent and shared cumulative ledger."""
import sys,json,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'result/rr_rank_v1'
PARENT=ROOT/'result/rr_v1'
LEDGER=PARENT/'gpu_budget.json'
PARENT_DIGEST='8b4311f96e7c5ce7391c24061c20ba60899779bd3fe1da21a4a794985d710919'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def source():
    rows=[json.loads(x) for x in (ROOT/'outputs/manifests/source_full.jsonl').read_text().splitlines()]
    if len(rows)!=10821 or len({r['category_id'] for r in rows})!=12 or sum(r['label'] for r in rows)!=1200:raise ValueError('Source contract changed')
    return rows
def parent_check():
    p=read(ROOT/'provenance/frozen_inputs.json')
    for rel,h in p['inputs'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Frozen input changed: '+rel)
    for rel,h in p['unchanged_science'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Unchanged science changed: '+rel)
    for rel,h in p['original_code'].items():
        if sha(Path(p['original_root'])/rel)!=h:raise ValueError('448 original changed: '+rel)
    return dict(files={**p['inputs'],**p['unchanged_science']},digest=p['original_contract_digest'])
def contract():
    files=dict(parent_check()['files'])
    for folder in ('src','code/rr_rank_v1','code/res518_v1','code/target518_v1','code/target14_v1'):
        for p in sorted((ROOT/folder).rglob('*.py')):files[str(p.relative_to(ROOT))]=sha(p)
    for rel in ('configs/base.yaml','configs/rr_rank_v1.json','configs/resolution518.json','docs/RESOLUTION_518_PROTOCOL_V1.md','provenance/backbone_weight.json','provenance/frozen_inputs.json','provenance/budget_authorization.json'):
        files[rel]=sha(ROOT/rel)
    return dict(files=files,digest=digest(files),parent=parent_check()['digest'],backbone='dinov2_vitl14',resolution=518)

def episode_identity(e):
    return dict(category=e['category'],cycle=e['cycle'],phase=e['phase'],k=e['k'],coherent=e['coherent'],
                query=[r['sample_id'] for r in e['query']],contexts=[[r['sample_id'] for r in c] for c in e['contexts']])

class SourceCache:
    def __init__(self,device):
        import yaml
        from src.cache import FeatureCache,cache_header
        self.device=device;self.header=cache_header(yaml.safe_load((ROOT/'configs/base.yaml').read_text()))
        self.caches=[FeatureCache(ROOT/'outputs/cache'/d,self.header,None,'cpu') for d in ('source_518',)]
    def path(self,row):
        for c in self.caches:
            p=c.path(row)
            if p.is_file():return c,p
        raise FileNotFoundError(row['sample_id'])
    def feature(self,row):return self.path(row)[0].read(row)
    def tensors(self,e):
        import torch
        from src.data.adapters import source_mask
        rows={r['sample_id']:r for r in e['query']+e['contexts'][0]+e['contexts'][1]}
        fs={k:self.feature(r) for k,r in rows.items()};ms={k:source_mask(r) for k,r in rows.items()}
        stack=lambda rs:torch.stack([fs[r['sample_id']] for r in rs]).to(self.device)
        masks=lambda rs:torch.stack([ms[r['sample_id']] for r in rs]).to(self.device)
        return dict(contexts=[stack(c) for c in e['contexts']],context_masks=[masks(c) for c in e['contexts']],query=stack(e['query']),
                    query_masks=masks(e['query']),labels=torch.tensor([r['label'] for r in e['query']],device=self.device),
                    sizes=[(r['height'],r['width']) for r in e['query']],category=e['category'])
