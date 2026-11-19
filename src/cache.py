"""Immutable four-layer cache keyed by complete extraction provenance."""
import hashlib
import json
import subprocess
from pathlib import Path
import torch
from .evidence import sha256,write_json
from .data.adapters import image_tensor


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def cache_header(config):
    from .models.backbone import PRESETS
    backbone_id=config['model']['backbone']
    if backbone_id not in PRESETS:raise ValueError('Unsupported backbone cache')
    dimension,layers=PRESETS[backbone_id]
    if tuple(config['model']['layer_indices_zero_based'])!=layers:raise ValueError('Wrong frozen layer selection')
    height,width=config['preprocess']['size']
    if height%14 or width%14:raise ValueError('Invalid patch resolution')
    ex=config['execution'];commit=subprocess.check_output(['git','-C',ex['backbone_repo'],'rev-parse','HEAD'],text=True).strip()
    return dict(backbone_id=backbone_id,weight_sha=sha256(ex['backbone_weights']),repo_commit=commit,
                layer_indices=list(layers),input_transform_hash=canonical_hash(config['preprocess']),
                extraction_precision='float32',dtype=config['cache']['feature_dtype'],shape=[4,height//14,width//14,dimension])


class FeatureCache:
    def __init__(self,root,header,backbone=None,device='cpu',microbatch=2):
        self.header=header;self.root=Path(root)/canonical_hash(header);self.root.mkdir(parents=True,exist_ok=True)
        self.backbone=backbone;self.device=device;self.microbatch=microbatch
        path=self.root/'header.json'
        if path.exists() and json.loads(path.read_text())!=header:raise ValueError('Cache header collision')
        if not path.exists():write_json(path,header)

    def path(self,row):
        return self.root/(canonical_hash({'header':self.header,'sample_sha':row['image_sha256']})+'.pt')

    def read(self,row):
        p=self.path(row);item=torch.load(p,map_location='cpu',weights_only=True)
        expected={**self.header,'sample_sha':row['image_sha256']}
        if item['metadata']!=expected:raise ValueError(f'Invalid cache metadata: {p}')
        x=item['features']
        if list(x.shape)!=expected['shape'] or x.dtype!=getattr(torch,expected['dtype']) or not torch.isfinite(x).all():raise ValueError('Invalid cache data')
        return x

    def ensure(self,rows):
        missing=[];seen=set()
        for row in rows:
            if row['image_sha256'] in seen:continue
            seen.add(row['image_sha256'])
            if not self.path(row).exists():missing.append(row)
        if missing and self.backbone is None:raise FileNotFoundError('Missing cache and no backbone')
        for i in range(0,len(missing),self.microbatch):
            batch=missing[i:i+self.microbatch]
            for row in batch:
                if sha256(row['path'])!=row['image_sha256']:raise ValueError('Source bytes changed since audit')
            x=torch.stack([image_tensor(r['path'],size=self.header['shape'][1]*14) for r in batch]).to(self.device)
            with torch.no_grad():features=self.backbone(x).to(getattr(torch,self.header['dtype'])).cpu()
            for row,feature in zip(batch,features):
                path=self.path(row);tmp=path.with_suffix('.tmp')
                torch.save({'metadata':{**self.header,'sample_sha':row['image_sha256']},'features':feature.clone()},tmp)
                tmp.replace(path)
            if i%20==0:print(f'cache {min(i+self.microbatch,len(missing))}/{len(missing)}',flush=True)

    def batch(self,rows):
        self.ensure(rows)
        return torch.stack([self.read(r) for r in rows]).to(self.device)


def code_digest(root='.'):
    root=Path(root);files={}
    for folder in ('src','tools','tests'):
        for p in sorted((root/folder).rglob('*.py')):files[str(p.relative_to(root))]=sha256(p)
    return canonical_hash(files),files
