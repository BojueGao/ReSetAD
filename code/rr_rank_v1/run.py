"""Engineering/full source training of the fixed loss-only revision."""
from common import *
import argparse,time,signal
import numpy as np
import torch
from model import RR
from sampling import PairSampler
from losses import paired_step
from diagnostics import run_diagnostics
from src.engine import seed_all,optimizer_for,StageScheduler,state_hash,save_checkpoint,restore_checkpoint
CYCLE=6036;TOTAL=30180

def components(phase):
    seed_all(17);m=RR().cuda();m.set_stage(phase);m.train()
    s=PairSampler(source(),read(PARENT/'source_neighbors.json'));s.begin_cycle(phase)
    opt=optimizer_for(m);sch=StageScheduler(opt,CYCLE if phase=='warmup' else 4*CYCLE)
    return m,s,opt,sch

def engineering():
    frozen=contract();cpu=read(RUN/'engineering/cpu_acceptance_v2.json')
    if cpu['status']!='ready' or cpu['contract']!=frozen:raise ValueError('CPU engineering contract mismatch')
    if (RUN/'engineering/gpu_acceptance.json').exists():raise FileExistsError('Engineering already complete')
    plan=read(RUN/'training_plan.json');cache=SourceCache('cuda');trace=[];timings={};torch.cuda.reset_peak_memory_stats()
    for phase,count in [('warmup',24),('joint',200)]:
        m,s,opt,sch=components(phase)
        for i in range(count):
            start=time.monotonic();e=s.next();batch=cache.tensors(e);terms=paired_step(m,opt,sch,batch,phase,plan['eta'][e['category']]);torch.cuda.synchronize()
            elapsed=time.monotonic()-start;trace.append(dict(phase=phase,step=i+1,category=e['category'],k=e['k'],coherent=e['coherent'],seconds=elapsed,**terms))
            if i%10==0:print(f'ENGINEERING {phase} {i+1}/{count} seconds={elapsed:.3f} loss={terms["total"]:.6f}',flush=True)
        ts=[r['seconds'] for r in trace if r['phase']==phase][2:]
        timings[phase]=dict(mean=float(np.mean(ts)),p90=float(np.quantile(ts,.9)),tasks=count,seconds=ts)
    ck=RUN/'engineering/exact_cuda_step200.pt';start=time.monotonic();save_checkpoint(ck,m,opt,sch,s,dict(contract=frozen,fixture=True));save_seconds=time.monotonic()-start
    e=s.next();expected=paired_step(m,opt,sch,cache.tensors(e),'joint',plan['eta'][e['category']]);expectedhash=state_hash(m);expectedstate=s.state_dict()
    restore_checkpoint(ck,m,opt,sch,s,dict(contract=frozen,fixture=True));e2=s.next();actual=paired_step(m,opt,sch,cache.tensors(e2),'joint',plan['eta'][e2['category']])
    if e!=e2 or expected!=actual or expectedhash!=state_hash(m) or expectedstate!=s.state_dict():raise AssertionError('CUDA exact restore failed')
    tasks=[t for t in read(RUN/'diagnostic_manifest.json')['tasks'] if t['k']==8 and t['style']=='coherent']
    if len(tasks)!=12:raise ValueError('Engineering diagnostic inventory')
    diag=run_diagnostics(m,cache,tasks,RUN/'engineering/diagnostics','engineering_only_not_scientific')
    updates=(timings['warmup']['mean']*CYCLE+timings['joint']['mean']*4*CYCLE)/3600
    # Include actual serialization rate for immutable250-step and cycle checkpoints.
    saves=len(set([0,TOTAL]+list(range(250,TOTAL+1,250))+list(range(CYCLE,TOTAL+1,CYCLE))))
    estimate=updates+save_seconds*saves/3600
    diagnostic_estimate=float(np.mean(diag['seconds_per_task']))*144/3600
    if contract()!=frozen:raise ValueError('Engineering files changed during execution')
    result=dict(status='ready',contract=frozen,at=now(),timings=timings,trace=trace,parameters=sum(p.numel() for p in m.parameters()),
        exact_cuda_recovery=True,checkpoint_sha256=sha(ck),checkpoint_save_seconds=save_seconds,planned_checkpoint_count=saves,
        measured_training_gpu_hours=estimate,conservative_training_gpu_hours=estimate*1.25+.5,
        measured_diagnostic_gpu_hours=diagnostic_estimate,conservative_diagnostic_gpu_hours=diagnostic_estimate*1.25+.1,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        diagnostic_receipt_sha256=sha(RUN/'engineering/diagnostics/receipt.json'),boundary='Engineering weights only; no effect or reference-purity improvement claim')
    write(RUN/'engineering/gpu_acceptance.json',result)
    print('GPU ENGINEERING READY '+str({k:result[k] for k in ('measured_training_gpu_hours','conservative_training_gpu_hours','measured_diagnostic_gpu_hours','conservative_diagnostic_gpu_hours','peak_allocated_bytes','peak_reserved_bytes')}),flush=True)

def train(resume=None):
    frozen=contract();accept=read(RUN/'engineering/gpu_acceptance.json');plan=read(RUN/'training_plan.json')
    if accept['status']!='ready' or accept['contract']!=frozen:raise ValueError('Engineering contract mismatch')
    phase='warmup';step0=0;m,s,opt,sch=components(phase);cache=SourceCache('cuda')
    if resume:
        latest=read(RUN/'latest_checkpoint.json')
        if str(Path(resume).resolve())!=str(Path(latest['path']).resolve()) or sha(resume)!=latest['sha256']:raise ValueError('Exact latest resume only')
        meta=torch.load(resume,map_location='cpu',weights_only=False)['metadata'];step0=meta['step'];phase=meta['phase']
        m.set_stage(phase);opt=optimizer_for(m);sch=StageScheduler(opt,CYCLE if phase=='warmup' else 4*CYCLE)
        restore_checkpoint(resume,m,opt,sch,s,dict(contract=frozen))
    else:write(RUN/'training_manifest.json',dict(contract=frozen,initialized_from_scratch=True,initial_model_hash=state_hash(m),at=now(),total_updates=TOTAL))
    stops=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda sig,frame:stops.append(sig))
    def save(step):
        path=RUN/'checkpoints'/f'step_{step:06d}.pt';save_checkpoint(path,m,opt,sch,s,dict(contract=frozen,step=step,phase=phase,cycle=s.cycle))
        write(RUN/'latest_checkpoint.json',dict(path=str(path),sha256=sha(path),step=step,phase=phase,at=now()),replace=True)
    if not resume:save(0)
    with (RUN/f'episodes_from_{step0:06d}_{time.time_ns()}.jsonl').open('x',buffering=1) as log:
        for step in range(step0+1,TOTAL+1):
            if s.position==len(s.schedule):
                if phase=='warmup':phase='joint';m.set_stage(phase);opt=optimizer_for(m);sch=StageScheduler(opt,4*CYCLE)
                s.begin_cycle(phase)
            e=s.next();identity=digest(episode_identity(e))
            if identity!=plan['task_digests'][step-1]:raise ValueError('Training task differs from frozen source plan')
            start=time.monotonic();terms=paired_step(m,opt,sch,cache.tensors(e),phase,plan['eta'][e['category']]);torch.cuda.synchronize()
            row=dict(step=step,phase=phase,cycle=s.cycle,category=e['category'],k=e['k'],coherent=e['coherent'],task_digest=identity,
                     query_ids=[r['sample_id'] for r in e['query']],query_labels=[r['label'] for r in e['query']],context_ids=[[r['sample_id'] for r in c] for c in e['contexts']],seconds=time.monotonic()-start,**terms)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if step%10==0 or step==1:print(f'{phase} step={step}/{TOTAL} cycle={s.cycle} total={terms["total"]:.6f}',flush=True)
            if step%CYCLE==0:write(RUN/'coverage'/f'cycle_{s.cycle}.json',s.audit_coverage())
            if step%250==0 or step%CYCLE==0 or stops:save(step)
            if stops:raise SystemExit(130)
    if sch.steps!=24144 or contract()!=frozen:raise AssertionError('Final scheduler/contract mismatch')
    export=RUN/'inference_model.pt'
    if export.exists():raise FileExistsError(export)
    torch.save(dict(model=m.state_dict(),contract=frozen,step=TOTAL,seed=17,state_hash=state_hash(m)),export)
    write(RUN/'training_complete.json',dict(status='TRAINING_COMPLETE_AWAITING_POSTFLIGHT',step=TOTAL,export_sha256=sha(export),at=now(),contract=frozen))

def final_diagnostic():
    frozen=contract();post=read(RUN/'postflight.json');done=read(RUN/'training_complete.json')
    if post['status']!='READY_FOR_DELIVERY_REVIEW' or done['contract']!=frozen:raise ValueError('Full training not accepted')
    if sha(RUN/'inference_model.pt')!=done['export_sha256']:raise ValueError('Final checkpoint changed')
    export=torch.load(RUN/'inference_model.pt',map_location='cpu',weights_only=False)
    m=RR().cuda();m.load_state_dict(export['model']);run_diagnostics(m,SourceCache('cuda'),read(RUN/'diagnostic_manifest.json')['tasks'],RUN/'diagnostics_final','final_source_mechanism')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['engineering','train','diagnostics'],required=True);p.add_argument('--resume');a=p.parse_args()
    if a.mode=='engineering':engineering()
    elif a.mode=='train':train(a.resume)
    else:final_diagnostic()
