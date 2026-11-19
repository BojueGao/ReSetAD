"""Annotations may be joined only after verifying a complete sealed prediction set."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageOps
from .evidence import sha256
from .metrics import ranking,aupro,localization
from sklearn.metrics import f1_score


def evaluate_sealed(directory,annotations):
    root=Path(directory);seal=json.loads((root/'seal.json').read_text())
    if not seal.get('sealed') or seal['status']!='READY':raise ValueError('Predictions not sealed/complete')
    if sha256(root/'predictions.json')!=seal['inventory_sha256']:raise ValueError('Sealed inventory changed')
    predictions=json.loads((root/'predictions.json').read_text());ann={r.sample_id:r for r in annotations}
    if len(ann)!=len(annotations) or set(ann)!={r['sample_id'] for r in predictions}:raise ValueError('Annotation/prediction coverage mismatch')
    results={}
    for pool in sorted({r['pool_id'] for r in predictions}):
        items=[r for r in predictions if r['pool_id']==pool];ys=[];scores=[];masks=[];maps=[];loc=[]
        for row in items:
            annotation=ann[row['sample_id']];ys.append(annotation.y);scores.append(row['image_score'])
            path=root/row['map_file']
            if sha256(path)!=row['map_sha256']:raise ValueError('Sealed pixel map changed')
            score=np.load(path)['pixel_map']
            if score.dtype!=np.float32:raise ValueError('Pixel scores must remain FP32')
            if annotation.mask_path is None:
                if annotation.y==0 and annotation.annotation_kind=='binary_anomaly_mask':mask=np.zeros(score.shape,dtype=bool)
                else:continue
            else:
                with Image.open(annotation.mask_path) as f:mask=np.asarray(ImageOps.exif_transpose(f))!=0
            if mask.shape!=score.shape:raise ValueError('Original mask/prediction geometry mismatch')
            masks.append(mask);maps.append(score)
            if annotation.y==1:loc.append(localization(mask,score))
        result={'image':ranking(ys,scores),'image_f1_050':float(f1_score(ys,np.asarray(scores)>=.5,zero_division=0))}
        if maps:
            result['pixel']=ranking(np.concatenate([v.ravel() for v in masks]),np.concatenate([v.ravel() for v in maps]))
            result['aupro']=aupro(masks,maps)
        else:result['pixel']={'status':'NA_NO_MASK','reason':'No pixel annotations','auroc':None,'ap':None}
        result['dice']=float(np.mean([v['dice'] for v in loc])) if loc else None
        result['iou']=float(np.mean([v['iou'] for v in loc])) if loc else None
        results[pool]=result
    return {'seal_sha256':sha256(root/'seal.json'),'pools':results,'total_classes':len(results)}
