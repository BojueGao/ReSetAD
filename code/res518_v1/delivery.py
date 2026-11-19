"""CPU-only final receipts and true source loss summaries; no new forward passes."""
import sys,json,csv,math,tarfile
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import *

def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)

def main():
    target=ROOT/'result/inference/large518_target12_v1';dest=ROOT/'result/delivery_518_v1';dest.mkdir(exist_ok=False)
    post=read(RUN/'postflight.json');diag=read(RUN/'diagnostics_final/postflight.json')
    assert post['actual_updates']==30180 and diag['tasks']==144
    entries=[]
    for p in RUN.glob('episodes_from_*.jsonl'):entries.extend(json.loads(s) for s in p.read_text().splitlines())
    assert len(entries)==30180 and sorted(x['step'] for x in entries)==list(range(1,30181))
    keys=('total','reference','pixel','rank','rank_raw','focal','dice','E_gap','grad_norm','seconds')
    def summary(rows,**meta):
        return dict(meta,steps=len(rows),**{k:sum(float(r[k]) for r in rows)/len(rows) for k in keys if all(k in r and isinstance(r[k],(int,float)) for r in rows)})
    cycles=[summary([r for r in entries if r['cycle']==c],cycle=c,phase='warmup' if c==0 else 'joint') for c in range(5)]
    classes=[summary([r for r in entries if r['category']==c],category=c) for c in sorted({r['category'] for r in entries})]
    csvout(dest/'loss_per_cycle.csv',cycles);csvout(dest/'training_per_class.csv',classes)
    csvout(dest/'loss_steps.csv',[{k:r[k] for k in ('step','cycle','phase','category',*keys) if k in r} for r in entries])
    b=read(LEDGER);assert all(a.get('seconds') is not None for a in b['attempts'])
    total=b['prior_gpu_hours']+sum(a['seconds']/3600 for a in b['attempts']);initial=read(ROOT/'provenance/initialization.json')['old_used']
    evidence={}
    for p in RUN.glob('jobs/*.json'):
        r=read(p)
        if r['name']=='518_delivery_evidence_v1':continue
        assert r['exit_code']==0 and r['ended_at'] and sha(r['log'])==r['log_sha256']
        evidence[str(p.relative_to(ROOT))]=sha(p)
    targets=[]
    for p in sorted((target/'postflight').glob('*.json')):
        r=read(p);assert r['status']=='ready' and r['all_raw_and_fused_sha_verified'];targets.append(r)
    assert len(targets)==12 and sum(r['raw_maps'] for r in targets)==31998 and sum(r['fused_maps'] for r in targets)==10666
    metrics=read(target/'metrics/metrics_per_dataset.json');assert len(metrics)==12
    receipt=dict(status='READY_FOR_DELIVERY_REVIEW',updates=30180,source_diagnostics=144,datasets=12,raw_maps=31998,fused_maps=10666,
                 cumulative_gpu_hours=total,this_experiment_gpu_hours=total-initial,remaining_gpu_hours=b['max_gpu_hours']-total,
                 prior_results_unchanged=True,ledger_sha256=sha(LEDGER),completed_job_receipts=evidence,
                 root_wrapper_exit='Must verify actual root job exit0 after this CPU stage returns',time=now())
    contract();write(dest/'acceptance.json',receipt)
    text=(target/'metrics/metrics_summary.md').read_text()
    text+='\nFull30180updates and144fixed source diagnostic pairs passed. All31998raw maps and10666fused maps have nativeFP32 SHA/geometry checks.\n'
    text+=f'\nGPUhours: experiment {total-initial:.6f}; cumulative {total:.6f}; remaining {b["max_gpu_hours"]-total:.6f}.\n'
    text+='\nSingle seed17; no target-selected checkpoint. Pixel metric fixtures do not constitute an independent full-dataset pixel-metric reimplementation. Final wrapper exit is checked separately.\n'
    (dest/'RESULTS.md').write_text(text,encoding='utf-8')
    chosen=[]
    for folder in (dest,target/'metrics',RUN/'jobs',ROOT/'provenance',ROOT/'docs'):
        for p in folder.rglob('*'):
            if p.is_file() and p.stat().st_size<32*1024**2 and p.suffix in ('.json','.csv','.md','.log'):chosen.append(p)
    chosen+=[LEDGER,RUN/'postflight.json',RUN/'latest_checkpoint.json',RUN/'diagnostics_final/postflight.json',RUN/'engineering/gpu_acceptance.json']
    chosen=sorted(set(chosen));manifest={str(p.relative_to(ROOT)):sha(p) for p in chosen}
    write(dest/'small_evidence_manifest.json',manifest)
    with tarfile.open(dest/'small_evidence.tar.gz','x:gz') as tar:
        for p in chosen:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        tar.add(dest/'small_evidence_manifest.json',arcname=str((dest/'small_evidence_manifest.json').relative_to(ROOT)))
    print('DELIVERY_READY',total,total-initial,flush=True)
if __name__=='__main__':main()
