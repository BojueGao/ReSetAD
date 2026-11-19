"""RR final-export adapter; frozen RR forward/grouping and frozen target metrics."""
import argparse, importlib.util, json, math, os, shutil, sys, time
from pathlib import Path
from collections import Counter, OrderedDict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code/target14_v1')]
from common import read,write,sha,now,csvwrite
from grouping import balanced_groups,rotating_turns
from src.engine import state_hash,seed_all
from src.evidence import write_json as atomic
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
rrmodel=module('rr_locked_model',ROOT/'code/rr_rank_v1/model.py')
sys.modules['model']=rrmodel
rrinfer=module('rr_locked_inference',ROOT/'code/rr_rank_v1/inference.py')
TRAIN=ROOT/'result/rr_rank_v1'
RUN=ROOT/'result/inference/large518_target12_v1'
OLD=RUN
LEDGER=ROOT/'result/rr_v1/gpu_budget.json'
DATASETS=('mvtec_ad','mpdd','btad','dtd_synthetic','brain_mri','br35h','headct','isic','cvc_clinicdb','cvc_colondb','tn3k','kvasir_seg')
SEEDS=(101,202,303)
EXPORT_SHA=None
STATE_SHA=None
def records(ds):
    if ds not in DATASETS:raise ValueError('Unauthorized dataset; Real-IAD excluded')
    return read(RUN/'inputs'/f'{ds}.json')
def poll_stop():
    if (RUN/'STOP_GPU').exists():raise RuntimeError('Budget stop; preserve partial files')
def gate():
    global EXPORT_SHA,STATE_SHA
    f=read(RUN/'preflight.json')
    EXPORT_SHA=f['model_sha256'];STATE_SHA=f['model_state_sha256']
    for name,digest in f['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen file changed: '+name)
    if sha(TRAIN/'inference_model.pt')!=EXPORT_SHA:raise ValueError('Wrong final export')
    return f
class FeatureProvider:
    def __init__(self,index_path=None):
        self.index=read(index_path or (RUN/'cache_complete.json'));self.lru=OrderedDict();self.verified=set()
    def __call__(self,rows):
        import torch
        poll_stop();values=[]
        for r in rows:
            h=r['sha256']
            if h not in self.lru:
                entry=self.index['features'][h];p=Path(entry['path'])
                if h not in self.verified:assert sha(p)==entry['file_sha256'];self.verified.add(h)
                item=torch.load(p,map_location='cpu',weights_only=True)
                assert item['metadata']=={**self.index['header'],'sample_sha':h}
                f=item['features'];assert list(f.shape)==[4,37,37,1024] and f.dtype==torch.float32 and torch.isfinite(f).all()
                self.lru[h]=f
            self.lru.move_to_end(h);values.append(self.lru[h])
            while len(self.lru)>64:self.lru.popitem(last=False)
        return torch.stack(values).cuda()
def load_model():
    gate()
    import torch
    seed_all(17);p=torch.load(TRAIN/'inference_model.pt',map_location='cpu',weights_only=True)
    m=rrmodel.RR();m.load_state_dict(p['model'],strict=True)
    assert state_hash(m)==STATE_SHA;m.requires_grad_(False);return m.cuda().eval()
def seal(dest,entries,**meta):
    write(dest/'predictions.json',entries);write(dest/'seal.json',dict(sealed=True,status='READY',inventory_sha256=sha(dest/'predictions.json'),**meta))
def predict_raw(model,rows,provider,root):
    import numpy as np
    root=Path(root);root.mkdir(parents=True,exist_ok=False);before=state_hash(model)
    for seed in SEEDS:
        dest=root/'passes'/str(seed);dest.mkdir(parents=True);entries=[];groups=rrinfer.round_plan(rows,seed)
        for gi,g in enumerate(groups):
            poll_stop();outputs=rrinfer.predict_group(model,g,provider)
            for turn in rotating_turns(g):
                q=turn['query'][0];v=outputs[q['sample_id']];name=f'{len(entries):07d}.npz'
                np.savez(dest/name,pixel_map=v.numpy().astype(np.float32))
                entries.append(dict(sample_id=q['sample_id'],pool_id=q['pool_id'],image_score=float(rrmodel.image_score(v)),
                    map_file=name,map_sha256=sha(dest/name),group_ids=turn['group_ids'],reference_ids=[r['sample_id'] for r in turn['context']],
                    group_size=len(g),context_size=len(turn['context']),rotation=turn['rotation']))
            if (gi+1)%10==0 or gi+1==len(groups):
                atomic(root/'progress.json',dict(seed=seed,groups_done=gi+1,groups_total=len(groups),images_done=len(entries),time=now()))
                print('PREDICT',root.name,seed,gi+1,'/',len(groups),'images',len(entries),flush=True)
        assert Counter(e['sample_id'] for e in entries)==Counter(r['sample_id'] for r in rows)
        assert state_hash(model)==before
        seal(dest,entries,expected_queries=len(rows),predicted_queries=len(rows),state_hash_before=before,state_hash_after=before,group_seed=seed)
    assert all(p.grad is None for p in model.parameters())
    write(root/'raw_complete.json',dict(status='ready',raw_queries=3*len(rows),images=len(rows),before=before,after=state_hash(model),time=now()))
def estimated_hours(timings,datasets):
    assert {t['dataset'] for t in timings}==set(datasets) and len(timings)==len(datasets)
    measured=sum(t['seconds']/t['images']*datasets[t['dataset']]['images'] for t in timings)
    return dict(measured_prediction_hours=measured/3600,conservative_hours=(2*measured+30*len(timings))/3600+.25)
def run_prediction(ds):
    gate();m=load_model();p=FeatureProvider();predict_raw(m,records(ds),p,RUN/'predictions'/ds)
    write(RUN/'prediction_receipts'/f'{ds}.json',dict(status='ready',model_sha256=EXPORT_SHA,before=STATE_SHA,after=state_hash(m),images=len(records(ds)),raw_queries=3*len(records(ds)),verified_features=len(p.verified),time=now()))
def fuse_audit(ds):
    import numpy as np, torch
    root=RUN/'predictions'/ds;rows=records(ds);lookups=[];plans=[]
    for seed in SEEDS:
        p=root/'passes'/str(seed);s=read(p/'seal.json');assert s['sealed'] and s['status']=='READY'
        assert s['state_hash_before']==s['state_hash_after']==STATE_SHA and sha(p/'predictions.json')==s['inventory_sha256']
        entries=read(p/'predictions.json');assert Counter(e['sample_id'] for e in entries)==Counter(r['sample_id'] for r in rows)
        expected={t['query'][0]['sample_id']:t for g in rrinfer.round_plan(rows,seed) for t in rotating_turns(g)}
        for e in entries:
            t=expected[e['sample_id']];assert e['group_ids']==t['group_ids'] and e['reference_ids']==[r['sample_id'] for r in t['context']]
        lookups.append({e['sample_id']:e for e in entries});plans.append(expected)
    dest=root/'fused';dest.mkdir(exist_ok=False);fused=[]
    for i,r in enumerate(sorted(rows,key=lambda r:r['sample_id'])):
        maps=[];evidence=[];gs=[];rs=[]
        for seed,lookup in zip(SEEDS,lookups):
            e=lookup[r['sample_id']];p=root/'passes'/str(seed)/e['map_file'];assert sha(p)==e['map_sha256']
            with np.load(p,allow_pickle=False) as z:v=z['pixel_map'].copy()
            assert v.dtype==np.float32 and v.shape==(r['height'],r['width']) and np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
            maps.append(torch.from_numpy(v));evidence.append(dict(seed=seed,sha256=e['map_sha256']))
            gs.append(tuple(sorted(e['group_ids'])));rs.append(tuple(sorted(e['reference_ids'])))
        mean,score=rrmodel.fuse_maps(maps);v=mean.numpy()
        # Independent double-precision arithmetic checks numerical error; retain exact frozen FP32 fusion.
        expected=np.stack([m.numpy() for m in maps]).astype(np.float64).mean(0)
        assert np.max(np.abs(v-expected))<=2*np.finfo(np.float32).eps
        n=max(1,math.ceil(v.size*.01));assert abs(float(score)-float(np.partition(v.ravel(),v.size-n)[-n:].astype(np.float64).mean()))<=2e-6
        name=f'{i:07d}.npz';np.savez_compressed(dest/name,pixel_map=v,segmentation_mask=(v>=.5).astype(np.uint8))
        with np.load(dest/name,allow_pickle=False) as z:assert np.array_equal(z['pixel_map'],v)
        fused.append(dict(sample_id=r['sample_id'],pool_id=r['pool_id'],image_score=float(score),predicted_anomaly=bool(score>=.5),map_file=name,map_sha256=sha(dest/name),pass_count=3,pass_evidence=evidence,unique_member_sets=len(set(gs)),unique_reference_sets=len(set(rs))))
        if (i+1)%100==0:print('FUSE_AUDIT',ds,i+1,'/',len(rows),flush=True)
    seal(dest,fused,expected_queries=len(rows),predicted_queries=len(rows),actual_raw_queries=3*len(rows),state_hash_before=STATE_SHA,state_hash_after=STATE_SHA)
    write(RUN/'postflight'/f'{ds}.json',dict(status='ready',images=len(rows),raw_maps=3*len(rows),fused_maps=len(rows),all_raw_and_fused_sha_verified=True,
        independent_pixel_average=True,fusion_arithmetic='Frozen RR FP32 torch.stack.mean; independent FP64 check within 2 FP32 epsilon',
        exact_group_and_reference_plan=True,all_native_fp32=True,time=now()))
def evaluate_dataset(ds):
    gate();assert read(RUN/'prediction_receipts'/f'{ds}.json')['status']=='ready'
    fuse_audit(ds)
    from evaluate import evaluate
    evaluate(ds,RUN/'predictions'/ds/'fused',RUN/'metrics'/ds)
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['benchmark','cache','cache_audit','predict','evaluate','report'],required=True);p.add_argument('--dataset',choices=DATASETS);a=p.parse_args()
    import torch
    torch.set_num_threads(4 if a.mode in ('predict','benchmark','cache') else 1)
    if a.mode=='predict':run_prediction(a.dataset)
    elif a.mode=='evaluate':evaluate_dataset(a.dataset)
    else:
        import lifecycle
        lifecycle.run(a.mode,sys.modules[__name__])
if __name__=='__main__':main()
