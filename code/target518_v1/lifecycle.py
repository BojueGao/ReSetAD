"""Large-only cache and measured admission; original RR scientific forward is reused."""
import sys,time,json,csv,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def unique_rows(w):
    return list({r['sha256']:r for ds in w.DATASETS for r in w.records(ds)}.values())

def as_cache_row(r):return dict(path=r['path'],image_sha256=r['sha256'])

def cache(w,backbone=None):
    from target.cache import TargetCache
    return TargetCache(w.ROOT/'outputs/cache/target_518_s002_v1',w.read(w.RUN/'preflight.json')['cache_header'],backbone,'cuda' if backbone is not None else 'cpu',microbatch=1)

def extract(w,rows,receipt):
    import torch,yaml
    from src.models.backbone import FrozenBackbone
    from src.engine import seed_all
    cfg=yaml.safe_load((w.ROOT/'configs/base.yaml').read_text());seed_all(17)
    start=time.monotonic();m=FrozenBackbone(cfg['execution']['backbone_repo'],cfg['execution']['backbone_weights'],cfg['model']['backbone'],cfg['model']['layer_indices_zero_based']).cuda()
    assert not m.training and not any(p.requires_grad for p in m.parameters())
    c=cache(w,m);missing=[r for r in rows if not c.path(as_cache_row(r)).exists()]
    torch.cuda.reset_peak_memory_stats()
    for i in range(0,len(missing),2):
        w.poll_stop();c.ensure([as_cache_row(r) for r in missing[i:i+2]])
        if (i+2)%100==0 or i+2>=len(missing):
            w.atomic(w.RUN/'cache_progress.json',dict(done=min(i+2,len(missing)),total=len(missing),reused=len(rows)-len(missing),time=w.now()))
            print('CACHE',min(i+2,len(missing)),len(missing),flush=True)
    torch.cuda.synchronize()
    result=dict(status='ready',images=len(rows),extracted=len(missing),reused=len(rows)-len(missing),seconds=time.monotonic()-start,
                max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),time=w.now())
    w.write(receipt,result);del c,m;torch.cuda.empty_cache();return result

def make_index(w,rows,destination):
    c=cache(w)
    def one(r):
        row=as_cache_row(r);f=c.read(row);p=c.path(row)
        return r['sha256'],dict(path=str(p),file_sha256=w.sha(p))
    with ThreadPoolExecutor(max_workers=4) as pool:entries=dict(pool.map(one,rows))
    result=dict(status='ready',images=len(rows),header=c.header,features=entries,fp32_metadata_shape_finite_sha_verified=True,time=w.now())
    w.write(destination,result);return result

def benchmark(w):
    import torch
    f=w.gate();samples={ds:[r for r in w.records(ds) if r['sample_id'] in set(f['datasets'][ds]['benchmark_sample_ids'])] for ds in w.DATASETS}
    unique=list({r['sha256']:r for rows in samples.values() for r in rows}.values())
    extraction=extract(w,unique,w.RUN/'benchmark_cache_extraction.json')
    make_index(w,unique,w.RUN/'benchmark_cache_index.json')
    model=w.load_model();timings=[]
    for ds,rows in samples.items():
        provider=w.FeatureProvider(w.RUN/'benchmark_cache_index.json');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        w.predict_raw(model,rows,provider,w.RUN/'benchmark'/ds)
        torch.cuda.synchronize();seconds=time.monotonic()-start
        entry=dict(dataset=ds,images=len(rows),raw_queries=3*len(rows),seconds=seconds,seconds_per_raw_query=seconds/(3*len(rows)),
                   max_reserved_bytes=torch.cuda.max_memory_reserved(),max_allocated_bytes=torch.cuda.max_memory_allocated())
        timings.append(entry);print('BENCHMARK',json.dumps(entry),flush=True);del provider
    prediction=w.estimated_hours(timings,f['datasets'])
    cache_hours=2*extraction['seconds']/max(1,extraction['extracted'])*f['unique_byte_hashes']/3600
    w.write(w.RUN/'benchmark.json',dict(status='ready',timings=timings,**prediction,cache_sample=extraction,
        conservative_cache_hours=cache_hours,conservative_total_hours=prediction['conservative_hours']+cache_hours+.25,
        model_state_hash=w.state_hash(model),formula='2x per-dataset three-pass prediction/write + startups + 0.25h; 2x cache estimate + 0.25h cache startup; manager additionally reserves1h',time=w.now()))

def report(w):
    metrics=['image_auroc','image_ap','pixel_auroc','aupro_030'];categories=[];datasets=[]
    for ds in w.DATASETS:
        audit=w.read(w.ROOT/'result/dataset_audit'/ds/'audit.json');result=w.read(w.RUN/'metrics'/ds/'metrics.json')
        assert w.read(w.RUN/'postflight'/f'{ds}.json')['all_raw_and_fused_sha_verified']
        rows=result['results'];domain='industrial' if ds in w.DATASETS[:4] else 'medical'
        for row in rows:
            r=dict(row,domain=domain)
            for k in metrics:r[k+'_percent']=None if r[k] is None else 100*r[k]
            categories.append(r)
        d=dict(dataset=ds,domain=domain,images=sum(r['images'] for r in rows),classes=len(rows),status='ready')
        for k in metrics:
            values=[r[k] for r in rows if r[k] is not None];d[k]=float(np.mean(values)) if values else None
            d[k+'_percent']=100*d[k] if values else None;d[k+'_valid_classes']=len(values)
        datasets.append(d)
    def output(name,rows):
        fields=list(dict.fromkeys(k for r in rows for k in r));w.csvwrite(w.RUN/'metrics'/name,rows,fields)
    output('metrics_per_category.csv',categories);output('metrics_per_dataset.csv',datasets)
    w.write(w.RUN/'metrics/metrics_per_dataset.json',datasets)
    base={r['dataset']:r for r in csv.DictReader((w.RUN/'inputs/base_metrics_per_dataset.csv').open(encoding='utf-8-sig'))}
    comparison=[]
    for r in datasets:
        for k in metrics:
            b=base[r['dataset']].get(k);b=None if b in (None,'','N/A','null') else float(b)
            comparison.append(dict(dataset=r['dataset'],metric=k,Large448=b,Large518=r[k],delta_percentage_points=None if b is None or r[k] is None else 100*(r[k]-b)))
    output('comparison_with_Large448.csv',comparison)
    fmt=lambda x:'N/A' if x is None else f'{100*x:.2f}'
    text='# DINOv2-Large 518：12个目标集三轮集合推理结果\n\n固定VisA源训练最终检查点；未调参、未重训。排除Real-IAD及KSDD；VisA不作为自身源训练模型的测试集。\n\n'
    text+='| 数据集 | 图像数 | 图像AUROC | 图像AP | 像素AUROC | AUPRO@0.30 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in datasets:text+='|'+r['dataset']+'|'+str(r['images'])+'|'+'|'.join(fmt(r[k]) for k in metrics)+'|\n'
    text+='\n百分制展示；CSV同时保留0–1原值。指标按完整类别计算，再对有效类别宏平均；无同组指标平均。三轮原尺寸FP32概率图先平均，图像分数为该平均图最高1%像素均值。\n\n'
    text+='图像单标签集的AUROC/AP为N/A；BrainMRI/Br35H/HeadCT无真实病灶掩码，像素指标N/A。像素AUROC精确使用全部FP32像素；PRO为8连通区域等权、4096个全类别等距阈值、归一化AUPRO@0.30。仅PRO阈值网格近似，无像素/图像抽样。\n\n'
    text+='448对照为已封存DINOv2-Large RR-Rank最终模型，数据清单、分组种子、科学前向函数、像素融合和评价代码一致；仅分辨率及对应空间网格、源训练得到的参数不同。各数据集/指标差值见comparison_with_Large448.csv。单seed对比不等于显著性检验。\n'
    (w.RUN/'metrics/metrics_summary.md').write_text(text,encoding='utf-8')

def run(mode,w):
    w.gate()
    if mode=='benchmark':benchmark(w)
    elif mode=='cache':extract(w,unique_rows(w),w.RUN/'cache_extraction.json')
    elif mode=='cache_audit':make_index(w,unique_rows(w),w.RUN/'cache_complete.json')
    elif mode=='report':report(w)
    else:raise ValueError(mode)
