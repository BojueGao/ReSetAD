"""S002-approved deterministic image decoding; separate inference and annotation tables."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from PIL import Image,ImageOps
import torch

DECODER={'id':'vsi_ad_s002_rgb_v1','allowed_modes':['L','RGB','RGBA','P'],'palette':'expand to RGB','alpha':'discard alpha retaining embedded RGB; no composite','exif':'transpose before RGB conversion','resize':[518,518],'filter':'PIL BICUBIC','normalization':'ImageNet mean/std','augmentation':'none'}
ROOT=Path('/home/ubuntu/MyDisk/VSI-AD')
AUDIT=ROOT/'outputs/final_preflight_20261014_v1'
OUT=ROOT/'outputs/final_stage_v1/target_registry'

def read_rgb(path):
    with Image.open(path) as image:
        image.load()
        if image.mode not in DECODER['allowed_modes']:raise ValueError(f'Unexpected image mode {image.mode}')
        return ImageOps.exif_transpose(image).convert('RGB')

def image_tensor(path):
    im=read_rgb(path).resize((518,518),Image.Resampling.BICUBIC)
    x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)

def build():
    authorization=json.loads((ROOT/'outputs/final_stage_v1/authorization.json').read_text())
    assert 'alpha discarded' in authorization['S002']
    OUT.mkdir(parents=True,exist_ok=False)
    registry={};source={json.loads(l)['image_sha256'] for l in (ROOT/'outputs/manifests/source_full.jsonl').read_text().splitlines()}
    masks={r['sample_id']:r for r in json.loads((AUDIT/'btad_mask_resolution_v2.json').read_text())}
    for ds,expected in [('mvtec_ad',1725),('mpdd',458),('btad',741),('brain_mri',253),('br35h',3000),('headct',200)]:
        image_rows=[];annotations=[]
        if ds=='headct':
            raw=json.loads((AUDIT/'headct_official_inventory_v2.json').read_text())
            images=ROOT/'outputs/final_stage_v1/inputs/headct';images.mkdir(parents=True)
            with zipfile.ZipFile(AUDIT/'official_downloads/headct.zip') as archive:
                for row in raw:
                    path=images/Path(row['member']).name
                    with path.open('xb') as f:f.write(archive.read(row['member']))
                    row.update(sample_id=hashlib.sha256(('headct:'+row['member']).encode()).hexdigest()[:24],pool_id='headct',path=str(path),image_sha256=row['sha256'],label_for_audit_only=row['label'],mask_path=None)
        else:raw=[json.loads(l) for l in (AUDIT/f'{ds}_audit_inventory.jsonl').read_text().splitlines()]
        for row in raw:
            assert sha(row['path'])==row['image_sha256'] and row['image_sha256'] not in source
            image=read_rgb(row['path']);assert image.size==(row['width'],row['height'])
            # Explicit dataclass fields only: no labels, folder status, GT or disease fields.
            image_rows.append(dict(sample_id=row['sample_id'],path=row['path'],height=row['height'],width=row['width'],pool_id=row['pool_id'],sha256=row['image_sha256'],optional_case_id=None))
            mask=row.get('mask_path');mask_sha=row.get('mask_sha256')
            if ds=='btad' and row['label_for_audit_only']:
                mask=masks[row['sample_id']]['mask'];mask_sha=masks[row['sample_id']]['mask_sha256']
            if mask:assert sha(mask)==mask_sha
            annotations.append(dict(sample_id=row['sample_id'],y=row['label_for_audit_only'],mask_path=mask,mask_sha256=mask_sha,kind='binary_anomaly_mask' if ds in ('mvtec_ad','mpdd','btad') else 'image_label_only'))
        assert len(image_rows)==expected and len({r['sample_id'] for r in image_rows})==expected
        image_rows.sort(key=lambda r:r['sample_id']);annotations.sort(key=lambda r:r['sample_id'])
        write(OUT/f'{ds}_images.json',image_rows);write(OUT/f'{ds}_annotations.json',annotations)
        registry[ds]=dict(status='READY',queries=expected,images=str(OUT/f'{ds}_images.json'),images_sha256=sha(OUT/f'{ds}_images.json'),annotations=str(OUT/f'{ds}_annotations.json'),annotations_sha256=sha(OUT/f'{ds}_annotations.json'),pools=sorted({r['pool_id'] for r in image_rows}),case_metadata='Unavailable; no inferred patient/video identity',source_audit=str(AUDIT),inventory='test only for industrial; canonical original release for medical',pixel_metrics=ds in ('mvtec_ad','mpdd','btad'))
    write(OUT/'registry.json',dict(status='READY',decoder=DECODER,datasets=registry,group_seeds=[101,202,303],core_only=True,training_images_added=False,authorization_sha256=sha(ROOT/'outputs/final_stage_v1/authorization.json'),target_results_used=False))
    print(json.dumps(dict(status='READY',queries=sum(r['queries'] for r in registry.values()),registry_sha256=sha(OUT/'registry.json'),decoder=DECODER)))

if __name__=='__main__':build()
