"""VisA original-release adapter; no reuse of any prior experiment's split."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
import torch
import torch.nn.functional as F
from src.evidence import sha256, write_json
from .manifest import save_rows
from src.resolution import INPUT_SIZE

HOLDOUT = ('pcb4','capsules','cashew')
EXPECTED = {'pcb1':1004,'pcb2':1001,'pcb3':1006,'pcb4':1005,'capsules':602,
            'candle':1000,'macaroni1':1000,'macaroni2':1000,'cashew':500,
            'chewinggum':503,'fryum':500,'pipe_fryum':500}


def read_image(path):
    with Image.open(path) as original:
        original.load()
        if original.mode not in ('RGB','L'):
            raise ValueError(f'Unexpected image mode {original.mode}: {path}')
        return ImageOps.exif_transpose(original).convert('RGB')


def image_tensor(path,size=INPUT_SIZE):
    image = read_image(path).resize((size,size), Image.Resampling.BICUBIC)
    x = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def source_mask(row,size=INPUT_SIZE):
    if row['label']==0:
        return torch.zeros(size,size)
    with Image.open(row['mask_path']) as f:
        m=np.asarray(ImageOps.exif_transpose(f)).copy()
    if m.ndim!=2: raise ValueError('Expected scalar source mask')
    m=torch.from_numpy(m!=0).float()[None,None]
    m=(F.interpolate(m,size=(size,size),mode='area')[0,0]>0).float()
    if not m.any():raise ValueError(f"Empty resized anomaly mask: {row['sample_id']}")
    return m


def phash(image):
    small=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
    values=dctn(small,type=2,norm='ortho')[:8,:8].ravel()
    bits=values>np.median(values[1:])
    return int.from_bytes(np.packbits(bits).tobytes(),'big')


def build_source(root, out):
    root=Path(root).resolve();out=Path(out)
    csv_path=root/'split_csv/1cls.csv'
    with csv_path.open(newline='',encoding='utf-8') as f: records=list(csv.DictReader(f))
    rows=[];seen_paths=set();errors=[]
    for raw in records:
        rel=raw['image'];cat=raw['object']
        path=(root/rel).resolve()
        path.relative_to(root)
        if rel in seen_paths: errors.append({'duplicate_original_id':rel});continue
        seen_paths.add(rel)
        if cat not in EXPECTED: raise ValueError(f'Unexpected category {cat}')
        if raw['label'] not in ('normal','anomaly'):raise ValueError(raw['label'])
        label=int(raw['label']=='anomaly');im=read_image(path);width,height=im.size
        mask_path=(root/raw['mask']).resolve() if label else None
        pixels=0;mask_hash=None
        if mask_path:
            mask_path.relative_to(root)
            with Image.open(mask_path) as mf: mask=np.asarray(ImageOps.exif_transpose(mf)).copy()
            if mask.ndim!=2 or mask.shape!=(height,width):raise ValueError(f'Mask shape mismatch {rel}')
            pixels=int((mask!=0).sum());mask_hash=sha256(mask_path)
            if pixels==0:raise ValueError(f'Empty anomaly mask {rel}')
        rows.append(dict(sample_id=hashlib.sha256(('visa:'+rel).encode()).hexdigest()[:24],
                         relative_path=rel,path=str(path),image_sha256=sha256(path),
                         category_id=cat,raw_category=cat,original_split=raw['split'],label=label,
                         mask_path=str(mask_path) if mask_path else None,mask_sha256=mask_hash,
                         height=height,width=width,mask_positive_pixels=pixels,phash=f'{phash(im):016x}'))
        if len(rows)%500==0:print(f'audited {len(rows)}/{len(records)}',flush=True)
    rows.sort(key=lambda r:r['sample_id'])
    counts=Counter((r['category_id'],r['label']) for r in rows)
    actual={c:{'normal':counts[c,0],'anomaly':counts[c,1]} for c in EXPECTED}
    for c,n in EXPECTED.items():
        if counts[c,0]!=n or counts[c,1]!=100:errors.append({'count_mismatch':c,'actual':actual[c],'expected_normal':n,'expected_anomaly':100})
    hashes=defaultdict(list)
    for r in rows:hashes[r['image_sha256']].append(r)
    exact=[[r['sample_id'] for r in group] for group in hashes.values() if len(group)>1]
    for group in hashes.values():
        if len({r['category_id'] in HOLDOUT for r in group})>1:
            errors.append({'train_holdout_hash_overlap':[r['sample_id'] for r in group]})
    # Near duplicates only reported, never deleted or used to change grouping by label.
    values=np.array([int(r['phash'],16) for r in rows],dtype=np.uint64)
    lookup=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
    near=[]
    for i,value in enumerate(values):
        xor=np.bitwise_xor(value,values[i+1:])
        distances=lookup[xor.view(np.uint8).reshape(-1,8)].sum(1)
        for offset in np.flatnonzero(distances<=4):
            near.append({'a':rows[i]['sample_id'],'b':rows[i+1+int(offset)]['sample_id'],'hamming':int(distances[offset])})
    out.mkdir(parents=True,exist_ok=True)
    save_rows(out/'source_full.jsonl',rows)
    save_rows(out/'source_dev_train.jsonl',[r for r in rows if r['category_id'] not in HOLDOUT])
    save_rows(out/'source_dev_holdout.jsonl',[r for r in rows if r['category_id'] in HOLDOUT])
    save_rows(out/'source_near_duplicates.jsonl',near)
    archives=list(root.parent.glob('*VisA*.tar*'))+list(root.parent.glob('*visa*.tar*'))
    audit=dict(status='READY' if not errors else 'INCOMPLETE',root=str(root),
               official_metadata_sha256=sha256(csv_path),total=len(rows),counts=actual,
               errors=errors,exact_duplicate_groups=exact,near_duplicate_pairs=len(near),
               archive_hashes={str(p):sha256(p) for p in archives if p.is_file()},
               archive_note='Archive absent: file-level SHA256 inventory retained' if not archives else None,
               manifest_sha256=sha256(out/'source_full.jsonl'))
    write_json(out/'source_audit.json',audit)
    return audit
