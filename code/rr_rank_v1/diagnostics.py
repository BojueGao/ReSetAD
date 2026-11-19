"""Fixed source mechanism audit. Predictions are sealed before annotation joins."""
from common import *
import numpy as np
import torch
import torch.nn.functional as F
from src.losses import normality_targets
from src.engine import state_hash

def query_targets(masks):
    patch=F.max_pool2d(masks[:,None].float(),14,14);parts=[]
    for r in (1,3,5):
        support=F.max_pool2d(F.pad(patch,(r//2,)*4,mode='reflect'),r,1) if r>1 else patch
        parts.append((1-support[:,0]).flatten(1))
    return torch.stack(parts,-1).repeat(1,1,4)

def scalar_auc(scores,normal):
    # Independent average-tie ranks; high reliability means normal.
    x=np.asarray(scores).ravel();y=np.asarray(normal,dtype=bool).ravel();pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return None
    order=np.argsort(x,kind='stable');ranks=np.empty(len(x),dtype=np.float64);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        ranks[order[i:j]]=(i+j+1)/2;i=j
    return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))

def selected_arrays(weights,matches,reference_images):
    w=np.asarray(weights);m=np.asarray(matches);r=np.asarray(reference_images)
    if m.ndim!=4 or r.shape!=m.shape[:-1]+(2,) or w.ndim!=3:raise ValueError('Invalid diagnostic shapes')
    if m.dtype.kind not in 'iu' or r.dtype.kind not in 'iu':raise ValueError('Noninteger diagnostic indices')
    if m.shape[-1]!=len(w) or m.shape[2]!=w.shape[2]:raise ValueError('Reference/branch mismatch')
    if r.min()<0 or r.max()>=len(w) or m.min()<0 or m.max()>=w.shape[1]:raise ValueError('Index out of range')
    if np.any(r[...,0]==r[...,1]):raise ValueError('Matched references must be distinct')
    anchors=np.take_along_axis(m,r,axis=-1);branch=np.arange(w.shape[2])[None,None,:,None]
    return anchors,w[r,anchors,branch]

def summarize_arrays(weights,matches,refs,context_normal,query_normal):
    w=np.asarray(weights);cn=np.asarray(context_normal);qn=np.asarray(query_normal)
    if w.shape!=cn.shape or not np.isfinite(w).all() or np.any((w<0)|(w>1)):raise ValueError('Invalid reliability values')
    if not np.isin(cn,[0,1]).all() or not np.isin(qn,[0,1]).all():raise ValueError('Nonbinary diagnostic annotation')
    anchors,selected_weight=selected_arrays(w,matches,refs)
    if qn.shape!=refs.shape[:-1]:raise ValueError('Query branch annotation mismatch')
    branch=np.arange(w.shape[2])[None,None,:,None];selected_bad=1-cn[refs,anchors,branch]
    high=w>=.5;total=float(w.sum(dtype=np.float64));bad=float((w*(1-cn)).sum(dtype=np.float64))
    result=dict(normality_auroc=scalar_auc(w,cn),normality_auroc_na_reason='single_anchor_label' if len(np.unique(cn))<2 else None,
        reference_anomaly_mass=bad/total if total else None,reference_weight_sum=total,reference_anomaly_weight_sum=bad,
        reference_anomaly_mass_na_reason=None if total else 'zero_reference_weight_sum',
        high_weight_normal_purity=float(cn[high].mean()) if high.any() else None,high_weight_count=int(high.sum()),
        high_weight_purity_na_reason=None if high.any() else 'no_reference_weight_ge_0.5',
        normal_weight_mean=float(w[cn==1].mean()) if (cn==1).any() else None,
        anomaly_weight_mean=float(w[cn==0].mean()) if (cn==0).any() else None,
        selected_weight_mean=float(selected_weight.mean()),selection={})
    for name,mask in [('all',np.ones_like(qn,dtype=bool)),('query_normal_patch',qn==1),('query_anomaly_patch',qn==0)]:
        values=selected_bad[mask];count=int(values.size);badcount=int(values.sum())
        result['selection'][name]=dict(anomaly_count=badcount,count=count,anomaly_fraction=badcount/count if count else None,na_reason=None if count else 'no_query_patches_in_stratum')
    return result

def context_gaps(records,a,b):
    if not np.isfinite(a) or a<=1e-6:raise ValueError('Invalid diagnostic readout slope')
    pairs={}
    for rec in records:pairs.setdefault((rec['task_id'],rec['variant']),{})[rec['view']]=rec
    reports=[]
    for (task_id,variant),views in sorted(pairs.items()):
        if set(views)!={0,1} or views[0]['query_ids']!=views[1]['query_ids']:raise ValueError('Incomplete diagnostic pair')
        scores=np.asarray([views[v]['scores'] for v in (0,1)],dtype=np.float64)
        if ((scores<=0)|(scores>=1)).any():raise ValueError('Saturated diagnostic score')
        energy=(np.log(scores)-np.log1p(-scores)-b)/a
        positive=np.asarray(views[0]['query_labels'],dtype=bool);negative=~positive;mixed=positive.any() and negative.any()
        worst=lambda x:float((x[:,positive].min(0)[None,:]-x[:,negative].max(0)[:,None]).mean()) if mixed else None
        reports.append(dict(task_id=task_id,variant=variant,category=views[0]['category'],k=views[0]['k'],style=views[0]['style'],
            E_by_context=energy.tolist(),mean_absolute_score_shift=float(np.abs(scores[1]-scores[0]).mean()),
            mean_absolute_E_shift=float(np.abs(energy[1]-energy[0]).mean()),worst_pair_score_gap=worst(scores),worst_pair_E_gap=worst(energy),
            pair_gap_na_reason=None if mixed else 'single_query_image_label'))
    return reports

@torch.inference_mode()
def run_diagnostics(model,cache,tasks,outdir,purpose):
    from src.data.adapters import source_mask
    outdir=Path(outdir)
    if outdir.exists():raise FileExistsError('Never overwrite a diagnostic attempt')
    outdir.mkdir(parents=True);rows={r['sample_id']:r for r in source()};before=state_hash(model);was_training=model.training;model.eval()
    records=[];times=[]
    import time
    try:
        for task in tasks:
            start=time.monotonic();qrows=[rows[s] for s in task['query']]
            q=torch.stack([cache.feature(r) for r in qrows]).to(cache.device)
            sizes=[(r['height'],r['width']) for r in qrows]
            for view,ids in enumerate(task['contexts']):
                crows=[rows[s] for s in ids];c=torch.stack([cache.feature(r) for r in crows]).to(cache.device)
                state=model.prepare_context(c)
                for variant in ('learned','uniform'):
                    st=state if variant=='learned' else {**state,'weights':torch.ones_like(state['weights'])}
                    out=model.score_queries(st,q,sizes)
                    raw=dict(weights=st['weights'].cpu().numpy(),matches=out['matches'].cpu().numpy(),reference_images=out['reference_images'].cpu().numpy(),
                             logits=out['logits'].cpu().numpy(),scores=out['scores'].cpu().numpy(),branch_distances=out['branch_distances'].cpu().numpy())
                    filename=f'{task["id"]}_v{view}_{variant}.npz';path=outdir/filename
                    with path.open('xb') as f:np.savez(f,**raw)
                    # Only now load labels/masks for metrics. No annotations in st/q.
                    cm=torch.stack([source_mask(r) for r in crows]);qm=torch.stack([source_mask(r) for r in qrows])
                    cn=normality_targets(cm).numpy();qn=query_targets(qm).numpy()
                    report=summarize_arrays(raw['weights'],raw['matches'],raw['reference_images'],cn,qn)
                    records.append(dict(task_id=task['id'],category=task['category'],k=task['k'],style=task['style'],view=view,variant=variant,
                        reference_image_ids=ids,query_ids=task['query'],query_labels=[r['label'] for r in qrows],scores=raw['scores'].tolist(),raw_file=filename,raw_sha256=sha(path),**report))
            if torch.cuda.is_available():torch.cuda.synchronize()
            times.append(time.monotonic()-start);print(f'DIAGNOSTIC {len(times)}/{len(tasks)} {task["id"]}',flush=True)
        after=state_hash(model)
        if before!=after:raise AssertionError('Diagnostic changed model state')
        a=float(F.softplus(model.rho));b=float(model.bias)
        result=dict(status='READY',purpose=purpose,scientific_result=purpose=='final_source_mechanism',source_training_overlap=True,
                    model_before=before,model_after=after,tasks=len(tasks),records=records,readout=dict(a=a,b=b),context_comparisons=context_gaps(records,a,b),
                    seconds_per_task=times,at=now(),boundary='Source controlled diagnostic, not target transfer; labels joined after sealed predictions; no optimizer')
        write(outdir/'receipt.json',result);return result
    finally:model.train(was_training)

def postflight(directory):
    from src.data.adapters import source_mask
    directory=Path(directory);receipt=read(directory/'receipt.json');rows={r['sample_id']:r for r in source()}
    if receipt['model_before']!=receipt['model_after']:raise ValueError('State mismatch')
    checked=0
    for rec in receipt['records']:
        p=directory/rec['raw_file']
        if sha(p)!=rec['raw_sha256']:raise ValueError('Diagnostic raw changed')
        with np.load(p,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        if raw['scores'].tolist()!=rec['scores']:raise ValueError('Score record mismatch')
        if any(not np.isfinite(x).all() for x in raw.values()):raise ValueError('Nonfinite trace')
        if any(raw[k].dtype!=np.float32 for k in ('weights','logits','scores','branch_distances')):raise ValueError('Trace must preserve FP32')
        cn=normality_targets(torch.stack([source_mask(rows[s]) for s in rec['reference_image_ids']])).numpy()
        qn=query_targets(torch.stack([source_mask(rows[s]) for s in rec['query_ids']])).numpy()
        # Scalar index lookup, independent of vectorized gather used for report.
        refs=raw['reference_images'];matches=raw['matches'];bad=0;count=0;strata={'query_normal_patch':[0,0],'query_anomaly_patch':[0,0]}
        for index in np.ndindex(refs.shape):
            qi,pi,bi,slot=index;ref=int(refs[index]);anchor=int(matches[qi,pi,bi,ref]);value=int(1-cn[ref,anchor,bi]);bad+=value;count+=1
            key='query_normal_patch' if qn[qi,pi,bi] else 'query_anomaly_patch';strata[key][0]+=value;strata[key][1]+=1
        if (bad,count)!=(rec['selection']['all']['anomaly_count'],rec['selection']['all']['count']):raise AssertionError('Scalar contamination mismatch')
        for key,(bad,count) in strata.items():
            if (bad,count)!=(rec['selection'][key]['anomaly_count'],rec['selection'][key]['count']):raise AssertionError('Stratum mismatch')
        fresh=summarize_arrays(raw['weights'],matches,refs,cn,qn)
        for key,value in fresh.items():
            if value!=rec[key]:raise AssertionError('Diagnostic aggregate mismatch: '+key)
        checked+=1
    if context_gaps(receipt['records'],**receipt['readout'])!=receipt['context_comparisons']:raise AssertionError('Context comparison mismatch')
    write(directory/'postflight.json',dict(status='READY',records=checked,tasks=receipt['tasks'],receipt_sha256=sha(directory/'receipt.json'),at=now(),independent_actual_index_lookup=True))
    print(f'DIAGNOSTIC POSTFLIGHT {checked} traces READY',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--postflight',required=True);a=p.parse_args();postflight(a.postflight)
