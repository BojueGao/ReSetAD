"""Independent evaluator metrics: full-resolution FP32, explicit NA reasons."""
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score


def ranking(labels,scores):
    y=np.asarray(labels,dtype=np.uint8);s=np.asarray(scores,dtype=np.float32)
    if y.shape!=s.shape or not np.isfinite(s).all():raise ValueError('Invalid prediction coverage/numerics')
    if len(np.unique(y))<2:
        return {'auroc':None,'ap':None,'status':'NA_ALL_POSITIVE' if y.all() else 'INCOMPLETE','reason':'Ranking needs both label states'}
    return {'auroc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s)),'status':'READY'}


def aupro(masks,maps,thresholds=4096,limit=.30):
    if len(masks)!=len(maps) or not maps:raise ValueError('Incomplete maps')
    scores=[];regions=[];background=[]
    for mask,score in zip(masks,maps):
        mask=np.asarray(mask,dtype=bool);score=np.asarray(score,dtype=np.float32)
        if mask.shape!=score.shape or not np.isfinite(score).all():raise ValueError('Geometry/numerics mismatch')
        ids,count=label(mask,structure=np.ones((3,3)))
        background.append(score[~mask]);scores.append(score)
        for k in range(1,count+1):regions.append(np.sort(score[ids==k]))
    if not regions:return {'value':None,'status':'NA_NO_MASK','reason':'No GT connected region'}
    bg=np.sort(np.concatenate(background))
    if not len(bg):return {'value':None,'status':'INCOMPLETE','reason':'No background pixels for FPR'}
    low=min(float(s.min()) for s in scores);high=max(float(s.max()) for s in scores)
    levels=np.linspace(low,high,thresholds,dtype=np.float64)
    fpr=(len(bg)-np.searchsorted(bg,levels,side='left'))/len(bg)
    pro=np.mean([(len(r)-np.searchsorted(r,levels,side='left'))/len(r) for r in regions],axis=0)
    # Include the empty prediction endpoint; keep maximum PRO at duplicate FPR.
    fpr=np.r_[0.,fpr];pro=np.r_[0.,pro];order=np.argsort(fpr,kind='stable');fpr=fpr[order];pro=pro[order]
    x,starts=np.unique(fpr,return_index=True);y=np.maximum.reduceat(pro,starts)
    upper=float(np.interp(limit,x,y));keep=x<limit;x=np.r_[x[keep],limit];y=np.r_[y[keep],upper]
    return {'value':float(np.trapezoid(y,x)/limit),'status':'READY','regions':len(regions),'thresholds':thresholds}


def localization(mask,score,threshold=.5):
    m=np.asarray(mask,dtype=bool);p=np.asarray(score)>=threshold
    overlap=int((m&p).sum());total=int(m.sum()+p.sum());union=int((m|p).sum())
    return {'dice':2*overlap/total if total else 1.,'iou':overlap/union if union else 1.}
