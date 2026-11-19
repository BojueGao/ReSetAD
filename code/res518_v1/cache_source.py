"""New Large source cache, FP32; no Base cache reuse and no target access."""
import sys,time,signal,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import *
import torch,yaml
from src.cache import cache_header,FeatureCache
from src.models.backbone import FrozenBackbone
from src.engine import seed_all

def build():
    cfg=yaml.safe_load((ROOT/'configs/base.yaml').read_text());header=cache_header(cfg)
    assert header['backbone_id']=='dinov2_vitl14' and header['shape']==[4,37,37,1024]
    assert header['weight_sha']==read(ROOT/'provenance/backbone_weight.json')['sha256']
    return cfg,header

def extract():
    cfg,header=build();seed_all(17);torch.set_num_threads(4)
    if read(RUN/'source_preflight.json')['status']!='READY':raise ValueError('Source not audited')
    model=FrozenBackbone(cfg['execution']['backbone_repo'],cfg['execution']['backbone_weights'],
                         cfg['model']['backbone'],cfg['model']['layer_indices_zero_based']).cuda()
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Encoder not frozen')
    cache=FeatureCache(ROOT/'outputs/cache/source_518',header,model,'cuda',microbatch=2)
    stops=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda sig,frame:stops.append(sig))
    rows=source();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i in range(0,len(rows),2):
        if stops:raise SystemExit(130)
        cache.ensure(rows[i:i+2])
        if (i+2)%100==0 or i+2>=len(rows):
            torch.cuda.synchronize();done=min(i+2,len(rows));elapsed=time.monotonic()-start
            write(RUN/'cache_progress.json',dict(done=done,total=len(rows),elapsed_seconds=elapsed,
                  seconds_per_image=elapsed/done,at=now()),replace=True)
            print('LARGE_CACHE',done,len(rows),f'{elapsed/done:.4f}s/image',flush=True)
    write(RUN/'cache_extraction.json',dict(status='READY',images=len(rows),header=header,
          seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_bytes=torch.cuda.max_memory_reserved(),at=now()))

def audit():
    cfg,header=build();cache=FeatureCache(ROOT/'outputs/cache/source_518',header)
    r=read(RUN/'cache_extraction.json');assert r['status']=='READY' and r['header']==header
    entries=[]
    for i,row in enumerate(source()):
        f=cache.read(row);p=cache.path(row)
        entries.append(dict(sample_id=row['sample_id'],sample_sha256=row['image_sha256'],path=str(p),sha256=sha(p)))
        if (i+1)%1000==0:print('CACHE_AUDIT',i+1,flush=True)
    write(RUN/'cache_complete.json',dict(status='READY',header=header,images=len(entries),entries=entries,
          fp32_shape_finite_and_file_sha_checked=True,at=now()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['extract','audit'],required=True)
    {'extract':extract,'audit':audit}[p.parse_args().mode]()
