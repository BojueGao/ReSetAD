"""Sealed full-category metrics; exact FP32 pixel AUROC, bounded-memory AUPRO."""
import sys,argparse,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from src.stream_metrics import ExactPixelRanks,StreamingPRO
from common import *
from audit import mask_array,SPECS

METRICS=['image_auroc','image_ap','pixel_auroc','aupro_030']
def image_metrics(labels,scores):
 labels=np.asarray(labels);scores=np.asarray(scores,dtype=np.float64)
 if labels.shape!=scores.shape or not np.isfinite(scores).all():raise ValueError('Invalid image ranking input')
 if set(labels.tolist())!={0,1}:return dict(image_auroc=None,image_ap=None,image_reason='N/A: binary evaluation requires both normal and abnormal images')
 return dict(image_auroc=float(roc_auc_score(labels,scores)),image_ap=float(average_precision_score(labels,scores)),image_reason=None)

def load_map(prediction,root,annotation,shape):
 p=(root/prediction['map_file']).resolve()
 if not p.is_relative_to(root.resolve()) or sha(p)!=prediction['map_sha256']:raise ValueError('Map outside seal or changed')
 with np.load(p,allow_pickle=False) as z:score=z['pixel_map'].copy()
 if score.dtype!=np.float32 or score.shape!=shape or not np.isfinite(score).all():raise ValueError('Invalid native FP32 map')
 if annotation['mask_path']:
  if sha(annotation['mask_path'])!=annotation['mask_sha256']:raise ValueError('Annotation changed')
  mask,_=mask_array(annotation['mask_path'])
  if mask.shape!=score.shape:raise ValueError('Native mask/map geometry mismatch; no silent resize')
 elif annotation['image_label']==0 and annotation['annotation_kind']=='binary_anomaly_mask':mask=np.zeros(shape,dtype=bool)
 else:mask=None
 return score,mask

def evaluate(ds,pred_root,output):
 pred_root=Path(pred_root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
 audit=read(RESULT/'dataset_audit'/ds/'audit.json')
 if audit['status']!='ready':raise ValueError('Dataset audit not ready')
 manifest=read(RESULT/'dataset_audit'/ds/'manifest.json')
 if sha(RESULT/'dataset_audit'/ds/'manifest.json')!=audit['manifest_sha256']:raise ValueError('Dataset manifest changed')
 ann={r['image_id']:r for r in manifest};seal=read(pred_root/'seal.json')
 if seal.get('status')!='READY' or not seal.get('sealed'):raise ValueError('Unsealed predictions')
 if seal['inventory_sha256']!=sha(pred_root/'predictions.json'):raise ValueError('Prediction manifest changed')
 if seal['state_hash_before']!=seal['state_hash_after']:raise ValueError('Target model state changed')
 predictions=read(pred_root/'predictions.json')
 if len(predictions)!=len(ann) or {r['sample_id'] for r in predictions}!=set(ann):raise ValueError('Prediction coverage must equal entire dataset')
 results=[]
 for category in sorted({r['category'] for r in manifest}):
  rows=[p for p in predictions if ann[p['sample_id']]['category']==category]
  if any(p['pool_id']!=category for p in rows):raise ValueError('Prediction category differs')
  entry=dict(dataset=ds,category=category,images=len(rows),status='ready',**image_metrics([ann[p['sample_id']]['image_label'] for p in rows],[p['image_score'] for p in rows]))
  low=math.inf;high=-math.inf;pixel_complete=True
  for p in rows:
   a=ann[p['sample_id']];sc,mask=load_map(p,pred_root,a,(a['height'],a['width']))
   # Classification is tied to the final fused segmentation evidence.
   n=max(1,math.ceil(sc.size*.01));expected=float(np.partition(sc.ravel(),sc.size-n)[-n:].astype(np.float64).mean())
   if abs(expected-p['image_score'])>2e-6:raise ValueError('Image score is not top1% of fused pixel probabilities')
   if mask is None:pixel_complete=False
   low=min(low,float(sc.min()));high=max(high,float(sc.max()))
  if not pixel_complete:
   entry.update(pixel_auroc=None,aupro_030=None,pixel_reason='N/A: real anomaly-region mask unavailable; no synthetic abnormal zero masks')
  else:
   rank=ExactPixelRanks(output/'scratch'/category);pro=StreamingPRO(low,high,thresholds=4096,limit=.30)
   try:
    for p in rows:
     a=ann[p['sample_id']];sc,mask=load_map(p,pred_root,a,(a['height'],a['width']));rank.add(mask,sc);pro.add(mask,sc)
    ranks=rank.result();region=pro.result();entry.update(pixel_auroc=ranks['auroc'],aupro_030=region['value'],pixel_reason=region.get('reason'),
        pixel_count=ranks['valid_pixels'],defect_regions=region.get('regions'),pixel_auroc_exact_fp32=True)
   finally:rank.close()
  results.append(entry)
 write(output/'metrics.json',dict(dataset=ds,results=results,seal_sha256=sha(pred_root/'seal.json'),manifest_sha256=audit['manifest_sha256'],
    metric_implementation=dict(source='original VSI-AD src/stream_metrics.py',sha256=sha(ROOT/'src/stream_metrics.py'),
      image_ap='sklearn.average_precision_score standard non-trapezoidal',pixel_auroc='exact FP32 sparse radix; all pixels including normal backgrounds',
      pro='normalized AUPRO@0.30',connectivity=8,thresholds=4096,threshold_method='linear full-category global score range; >=; exact per-image region histograms',
      integral='trapezoid over FPR with interpolated endpoint 0.30, divided by0.30; maxPRO at duplicate FPR',
      approximation='PRO threshold grid only; no image/pixel subsampling, no group min-max',statistical_unit='image; region equal-weight PRO'),time=now()))
 return results

def report():
 categories=[];datasets=[]
 for ds in SPECS:
  ap=RESULT/'dataset_audit'/ds/'audit.json';mp=RESULT/'metrics'/ds/'metrics.json'
  if not ap.exists():continue
  a=read(ap);computed=read(mp)['results'] if mp.exists() else []
  for c in a['category_counts']:
   match=next((x for x in computed if x['category']==c['category']),None)
   r=dict(dataset=ds,domain=SPECS[ds][0],category=c['category'],images=c['images'],normal=c['normal'],anomaly=c['anomaly'],
          status='NOT_RUN_RETRAIN_FIRST',reason='User clarified: retrain first; no target inference in this stage',**{k:None for k in METRICS})
   if match:r.update(match)
   r['image_applicability']='valid_binary' if c['normal'] and c['anomaly'] else 'N/A_SINGLE_IMAGE_LABEL'
   r['pixel_applicability']='N/A_NO_REAL_MASK' if ds in CORE_CLASSIFICATION else 'valid_if_all_true_masks'
   for k in METRICS:r[k+'_percent']=100*r[k] if r[k] is not None else None
   categories.append(r)
  rr=[r for r in categories if r['dataset']==ds];entry=dict(dataset=ds,domain=SPECS[ds][0],classes=len(rr),images=sum(r['images'] for r in rr),status='ready' if computed else 'NOT_RUN_RETRAIN_FIRST')
  for k in METRICS:
   values=[r[k] for r in rr if r[k] is not None];entry[k]=float(np.mean(values)) if values else None;entry[k+'_percent']=entry[k]*100 if values else None;entry[k+'_valid_classes']=len(values)
  datasets.append(entry)
 csvwrite(RESULT/'metrics/metrics_per_category.csv',categories,list(categories[0]))
 csvwrite(RESULT/'metrics/metrics_per_dataset.csv',datasets,list(datasets[0]))
 write(RESULT/'metrics/metrics_per_dataset.json',datasets)
 text='# 14目标数据集评价接入\n\n当前用户已明确先重新训练，不启动目标推理。因此本轮指标均未计算；空值代表 NOT_RUN，不是零分或已有结果。\n\n'
 text+='图像AUROC与标准AP需要两种图像标签；病灶全阳性集的两项为N/A。BrainMRI/Br35H/HeadCT没有真实病灶掩码，像素指标N/A。\n\n'
 text+='| 数据集 | 图像数 | 类别数 | 状态 |\n|---|---:|---:|---|\n'
 for r in datasets:text+=f'|{r["dataset"]}|{r["images"]}|{r["classes"]}|{r["status"]}|\n'
 text+='\n工业与医学分开统计；目前两域各指标的有效已评价数据集/类别数均为0。未来每类先全量计算，再对有效类别等权宏平均；数据集汇总继续等权，并报告分母。\n\n像素AUROC使用全部FP32像素精确排序；AUPRO@0.30使用8连通区域等权，4096个全类别分数范围等距阈值，端点线性插值并除以0.30。仅PRO阈值网格近似，无图像/像素抽样。\n'
 (RESULT/'metrics/metrics_summary.md').write_text(text,encoding='utf-8')
CORE_CLASSIFICATION={'brain_mri','br35h','headct'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--predictions');p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.report:report()
 else:evaluate(a.dataset,a.predictions,RESULT/'metrics'/a.dataset)
