"""Target-only inventory. Labels are written to a separate evaluation manifest."""
import sys, os, re, argparse, collections, concurrent.futures
import numpy as np
from PIL import Image, ImageOps
from common import *

IMAGE_EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
CORE=['mvtec_ad','mpdd','btad','brain_mri','br35h','headct']
SPECS={
 'mvtec_ad':('industrial','official_test',1725,'https://www.mvtec.com/research-teaching/datasets/mvtec-ad'),
 'mpdd':('industrial','author_official_test',458,'https://github.com/stepanje/MPDD'),
 'btad':('industrial','official_test',741,'https://avires.dimi.uniud.it/papers/btad/btad.zip'),
 'ksdd':('industrial','KolektorSDD_v1_fine_masks_full399',399,'https://www.vicos.si/resources/kolektorsdd/'),
 'realiad':('industrial','official_1024_supervised_test_image_unit',None,'https://realiad4ad.github.io/Real-IAD/'),
 'dtd_synthetic':('industrial','AnomalyCLIP_author_benchmark_test',None,'https://github.com/zqhang/AnomalyCLIP'),
 'brain_mri':('medical','navoneel_original_yes_no',253,'https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection'),
 'br35h':('medical','Br35H_2020_yes_no_exclude_pred',3000,'https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection'),
 'headct':('medical','felipekitamura_original_labeled200',200,'https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage'),
 'isic':('medical','ISIC2016_Part1_official_test',379,'https://challenge.isic-archive.com/data/'),
 'cvc_clinicdb':('medical','original612_AnomalyCLIP_format',612,'https://pages.cvc.uab.es/CVC-Colon/index.php/databases/'),
 'cvc_colondb':('medical','original_unaugmented380_AnomalyCLIP_format',380,'https://pages.cvc.uab.es/CVC-Colon/index.php/databases/'),
 'tn3k':('medical','official_test614_AnomalyCLIP_format',614,'https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation'),
 'kvasir_seg':('medical','Kvasir_SEG_original1000_author_linked_Polyp_v2',1000,'https://datasets.simula.no/kvasir-seg/'),
}

def mask_array(path):
 with Image.open(path) as im:
  im=ImageOps.exif_transpose(im);im.load();v=np.asarray(im.convert('L'))
 threshold=0 if int(v.max())<=1 else 127
 return v>threshold,dict(mask_threshold=threshold,mask_gray_levels=int(len(np.unique(v))),mask_size=list(v.shape))

def base_row(ds,category,image,mask,label,split,**extra):
 image=Path(image);uid=hashlib.sha256(f'{ds}:{image}'.encode()).hexdigest()[:24]
 return dict(dataset=ds,version=SPECS[ds][1],category=category,official_split=split,image_id=uid,
             image_path=str(image),image_label=label,mask_path=str(mask) if mask else None,
             mask_available=bool(mask),annotation_kind='image_label_only' if ds in CORE[3:] else 'binary_anomaly_mask',
             patient_id=None,instance_id=None,view_id=None,**extra)

def enumerate_rows(ds):
 notes=[];meta=[];raw_count=None
 if ds in CORE:
  reg=ROOT/'outputs/final_stage_v1/target_registry';images=reg/f'{ds}_images.json';anns=reg/f'{ds}_annotations.json'
  meta=[images,anns];annotation={x['sample_id']:x for x in read(anns)};rows=[]
  for x in read(images):
   a=annotation[x['sample_id']]
   r=base_row(ds,x['pool_id'],x['path'],a['mask_path'],a['y'],'test' if ds in CORE[:3] else 'published_release',prior_id=x['sample_id'],prior_sha256=x['sha256'])
   rows.append(r)
  notes.append('Reuses original source-verified core inventory; original images retained, S002 RGB decoding, no patient IDs invented.')
  return rows,notes,meta,raw_count
 if ds=='ksdd':
  root=DATA/'KolektorSDD_v1';rows=[]
  for image in sorted(root.glob('kos*/*.jpg')):
   mask=image.with_name(image.stem+'_label.bmp');y=int(mask_array(mask)[0].any())
   r=base_row(ds,'electrical_commutator',image,mask,y,'published_release');r['instance_id']=image.parent.name;r['view_id']=image.stem;rows.append(r)
  notes.append('Official fine annotation archive; all 399 source images used as target-only release, not official cross-validation training. item/view names preserved.')
  meta=[OTHER/'downloads/KolektorSDD_official.zip']
  return rows,notes,meta,len(rows)
 if ds=='realiad':
  root=DATA/'Real-IAD/realiad_1024';metaroot=DATA/'Real-IAD/realiad_jsons/realiad_jsons';rows=[];raw_count=0
  meta=sorted(metaroot.glob('*.json'))
  if len(meta)!=30:raise ValueError('Expected 30 official Real-IAD split JSONs')
  for m in meta:
   obj=read(m);category=m.stem;raw_count+=len(obj['train'])+len(obj['test'])
   for x in obj['test']:
    rel=Path(x['image_path']);image=root/category/rel;mask=root/category/x['mask_path'] if x.get('mask_path') else None
    r=base_row(ds,category,image,mask,int(x['anomaly_class']!=obj['meta']['normal_class']),'test')
    r['instance_id']=str(rel.parent);vm=re.search(r'_C([1-5])_',rel.name);r['view_id']='C'+vm.group(1) if vm else None
    r['official_anomaly_class']=x['anomaly_class'];rows.append(r)
  notes.append('Official supervised multiview JSON test only; 1024 publisher archive; no 256/512 replacement. Image is statistical unit; instance identity is category + original parent path; five camera view tokens retained. All train files excluded from evaluation.')
  return rows,notes,meta,raw_count
 roots={'dtd_synthetic':DATA/'DTD-Synthetic-unzipped','isic':DATA/'ISIC','cvc_clinicdb':DATA/'CVC-ClinicDB','cvc_colondb':DATA/'CVC-ColonDB',
        'tn3k':OTHER/'downloads/TN3K/TN3K','kvasir_seg':OTHER/'downloads/Polyp_verified/Kvasir'}
 root=roots[ds];rows=[]
 if ds=='kvasir_seg':
  for im in sorted((root/'images').iterdir()):
   if im.suffix.lower() not in IMAGE_EXT:continue
   mask=root/'masks'/im.name
   rows.append(base_row(ds,'polyp',im,mask,1,'published_release'))
  meta=[OTHER/'downloads/Polyp_verified_extraction.json',OTHER/'extension_audit_v1/kvasir_seg/receipt.json']
  notes.append('Reuses author-linked Figshare Polyp_DataSet v2 complete 1000 Kvasir-SEG frames; not full Kvasir/Hyper-Kvasir. No patient/video IDs in release; no IDs inferred from image names.')
 else:
  if (root/'meta.json').exists():meta=[root/'meta.json']
  for cat in sorted(x for x in root.iterdir() if x.is_dir()):
   for im in sorted((cat/'test').glob('*/*')):
    if not im.is_file() or im.suffix.lower() not in IMAGE_EXT:continue
    label=int(im.parent.name not in ('good','ok'))
    mask=None
    if label:
     folder=cat/'ground_truth'/im.parent.name
     matches=[p for p in folder.glob('*') if p.stem in (im.stem,im.stem+'_mask',im.stem+'_Segmentation') and p.suffix.lower() in IMAGE_EXT]
     if len(matches)!=1:raise ValueError(f'Ambiguous/missing mask: {im}: {matches}')
     mask=matches[0]
    rows.append(base_row(ds,cat.name,im,mask,label,'test' if ds in ('tn3k','isic','dtd_synthetic') else 'published_release'))
  if meta:
   declared={x['img_path'] for vs in read(meta[0]).get('test',{}).values() for x in vs}
   observed={str(Path(x['image_path']).relative_to(root)) for x in rows}
   notes.append(f'Metadata discrepancy: {len(observed-declared)} actual test images absent from meta; {len(declared-observed)} declared absent from disk. Inventory rebuilt from ALL on-disk test images/masks, no score-based exclusions.')
  raw_count=sum(1 for c in root.iterdir() if c.is_dir() for split in ('train','test') for f in (c/split).glob('*/*') if f.suffix.lower() in IMAGE_EXT)
  notes.append('No crops, renames, resizing or conversion on disk. JPEG/binary masks decoded to L and thresholded >127 (or >0 for 0/1 storage), with gray-level audit; no anatomical substitute masks.')
 return rows,notes,meta,raw_count

def inspect(row):
 r=dict(row);errors=[]
 try:
  path=Path(r['image_path']);r['image_sha256']=sha(path)
  if r.get('prior_sha256') and r['prior_sha256']!=r['image_sha256']:errors.append('prior_image_hash_changed')
  with Image.open(path) as im:
   r['original_mode']=im.mode;r['exif_orientation']=im.getexif().get(274)
   im=ImageOps.exif_transpose(im);im.load();im=im.convert('RGB');r['width'],r['height']=im.size
   r['decoded_rgb_sha256']=hashlib.sha256(im.size[0].to_bytes(4,'big')+im.size[1].to_bytes(4,'big')+im.tobytes()).hexdigest()
  if r['mask_path']:
   mask,info=mask_array(r['mask_path']);r.update(info);r['mask_sha256']=sha(r['mask_path']);r['positive_mask_pixels']=int(mask.sum())
   if mask.shape!=(r['height'],r['width']):errors.append('image_mask_geometry_mismatch')
   if r['image_label']==0 and mask.any():errors.append('normal_label_with_positive_mask')
  else:
   r.update(mask_sha256=None,positive_mask_pixels=None,mask_threshold=None)
   if r['image_label']==1 and r['annotation_kind']!='image_label_only':errors.append('missing_anomaly_mask')
 except Exception as e:errors.append(f'{type(e).__name__}: {e}')
 r['audit_errors']=errors;return r

def audit(ds):
 dest=RESULT/'dataset_audit'/ds
 if (dest/'audit.json').exists():return read(dest/'audit.json')
 rows,notes,meta,raw=enumerate_rows(ds)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  inspected=[]
  for r in pool.map(inspect,rows):
   inspected.append(r)
   if len(inspected)%2000==0:print(f'AUDIT {ds} {len(inspected)}/{len(rows)}',flush=True)
 rows=inspected
 if len({x['image_id'] for x in rows})!=len(rows):raise ValueError('Duplicate manifest IDs')
 issues=[{'image_id':r['image_id'],'errors':r['audit_errors']} for r in rows if r['audit_errors']]
 expected=SPECS[ds][2]
 if expected is not None and len(rows)!=expected:issues.append({'expected_count':expected,'actual_count':len(rows)})
 if ds=='realiad' and len({x['category'] for x in rows})!=30:issues.append({'missing_categories':True})
 if not rows:issues.append({'empty_inventory':True})
 duplicates={}
 for field in ('image_sha256','decoded_rgb_sha256'):
  g=collections.defaultdict(list)
  for r in rows:
   if r.get(field):g[r[field]].append(r['image_id'])
  duplicates[field]={k:v for k,v in g.items() if len(v)>1}
 counts=[]
 for cat in sorted({r['category'] for r in rows}):
  rr=[x for x in rows if x['category']==cat]
  counts.append(dict(category=cat,normal=sum(x['image_label']==0 for x in rr),anomaly=sum(x['image_label']==1 for x in rr),images=len(rr),
                     valid_anomaly_masks=sum(x['image_label']==1 and x['mask_available'] and not x['audit_errors'] for x in rr)))
 summary=dict(dataset=ds,domain=SPECS[ds][0],version=SPECS[ds][1],source=SPECS[ds][3],status='partial' if issues else 'ready',
  actual_paths=sorted({str(Path(r['image_path']).parent.parent) for r in rows}),raw_image_count=raw if raw is not None else len(rows),
  evaluation_image_count=len(rows),category_counts=counts,valid_anomaly_masks=sum(c['valid_anomaly_masks'] for c in counts),
  official_split=sorted({r['official_split'] for r in rows}),exclusions='train images excluded for official-test targets; pred excluded for Br35H; no corrupt/duplicate sample silently removed',
  processing=notes,issues=issues,metadata=[dict(path=str(m),sha256=sha(m)) for m in meta],
  duplicates={k:{'groups':len(v),'excess_ids':sum(len(a)-1 for a in v.values())} for k,v in duplicates.items()},
  patient_ids_available=any(r['patient_id'] for r in rows),instance_ids_available=any(r['instance_id'] for r in rows),completed_at=now())
 write(dest/'manifest.json',rows);write(dest/'duplicate_groups.json',duplicates)
 csvwrite(dest/'manifest.csv',rows,list(rows[0]) if rows else ['dataset','image_id'])
 # No labels, masks, diagnostic type, or status directories are exposed as model inputs.
 model_rows=[dict(sample_id=r['image_id'],path=r['image_path'],pool_id=r['category'],height=r.get('height'),width=r.get('width'),
                 sha256=r.get('image_sha256'),decoded_rgb_sha256=r.get('decoded_rgb_sha256'),
                 optional_case_id=hashlib.sha256(f"{ds}:{r['category']}:{r['patient_id'] or r['instance_id']}".encode()).hexdigest() if r['patient_id'] or r['instance_id'] else None,
                 view_id=r['view_id']) for r in rows]
 write(dest/'images.json',model_rows)
 write(dest/'annotations.json',[{k:r.get(k) for k in ('image_id','image_label','mask_path','mask_available','mask_sha256','mask_threshold','annotation_kind')} for r in rows])
 summary['manifest_sha256']=sha(dest/'manifest.json');summary['images_sha256']=sha(dest/'images.json');summary['annotations_sha256']=sha(dest/'annotations.json')
 write(dest/'audit.json',summary)
 (dest/'audit.md').write_text(f"# {ds}\n\n状态：{summary['status']}；实际评估 {len(rows)} 图。\n\n"+'\n\n'.join(notes)+f"\n\n来源：{SPECS[ds][3]}\n\n完整类别计数、逐文件哈希、标注配对与问题见 audit.json / manifest.json。原始文件未修改。\n",encoding='utf-8')
 print(f'AUDIT COMPLETE {ds} {len(rows)} {summary["status"]}',flush=True);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='all');a=p.parse_args();summaries=[]
 for ds in SPECS if a.dataset=='all' else [a.dataset]:
  try:summaries.append(audit(ds))
  except Exception as e:
   error=dict(dataset=ds,status='blocked',error=repr(e),time=now());print(json.dumps(error),flush=True)
   path=RESULT/'dataset_audit'/ds/'blocked.json'
   if not path.exists():write(path,error)
   summaries.append(error)
 if a.dataset=='all':
  output=[]
  for s in summaries:
   output.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in s.items()})
  csvwrite(RESULT/'dataset_audit/dataset_inventory.csv',output,['dataset','domain','version','actual_paths','source','raw_image_count','evaluation_image_count','category_counts','valid_anomaly_masks','official_split','processing','exclusions','duplicates','status','error'])
  write(RESULT/'dataset_audit/inventory.json',summaries)
if __name__=='__main__':main()
