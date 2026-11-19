"""Append-only historical attempts, bounded engineering, full-duration admission."""
from common import *
import subprocess,time,signal,fcntl,argparse,sys

def ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return None
def used(b):return b['prior_gpu_hours']+sum((a['seconds'] if a.get('seconds') is not None else time.time()-a['start_epoch'])/3600 for a in b['attempts'])
def load_ledger():
    prefix=read(RUN/'budget_prefix_snapshot.json')['ledger'];b=read(LEDGER)
    if b['prior_gpu_hours']!=prefix['prior_gpu_hours'] or b['attempts'][:len(prefix['attempts'])]!=prefix['attempts']:raise ValueError('Historical budget prefix changed')
    if b['max_gpu_hours']<85 or b['save_at_gpu_hours']!=b['max_gpu_hours']-.25:raise ValueError('Invalid ledger cap')
    if any(a.get('seconds') is None for a in b['attempts']):raise RuntimeError('Open allocation: inspect original PID/start_ticks; no automatic recovery')
    return b
def device_check():
    if subprocess.check_output(['hostname'],text=True).strip()!='ubuntu-MZF2-AC0-00':raise RuntimeError('Not A100-15')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(rows)!=4 or any('A100-SXM4-40GB' not in r for r in rows):raise RuntimeError('Wrong GPU host')
    free=[int(row.split(',')[0]) for row in rows if int(row.rsplit(',',1)[1])<100]
    if not free:raise RuntimeError('No free A100; do not touch other jobs')
    return rows,free[0]
def allocate(b,name,mode,hours):
    if any(a['name']==name for a in b['attempts']):raise FileExistsError('Attempt already exists; inspect it')
    if used(b)+hours>b['save_at_gpu_hours']:raise RuntimeError('Allocation cap does not fit remaining budget')
    devices,gpu=device_check();command=[sys.executable,'-u','code/rr_rank_v1/run.py','--mode',mode]
    log=RUN/'jobs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True);start=time.time()
    a=dict(name=name,run='rr_rank_v1',start_epoch=start,started_at=now(),seconds=None,ended_at=None,exit_code=None,command=command,gpu=gpu,device_snapshot=devices,
           boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),log=str(log))
    b['attempts'].append(a);write(LEDGER,b,replace=True);child=None;stops=[]
    try:
        with log.open('xb',buffering=0) as f:
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            a.update(pid=child.pid,start_ticks=ticks(child.pid));write(LEDGER,b,replace=True)
            for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda sig,frame:stops.append(sig))
            stop_time=None
            while child.poll() is None:
                u=used(b)
                if (stops or u>=b['save_at_gpu_hours'] or time.time()-start>=hours*3600) and stop_time is None:
                    os.killpg(child.pid,signal.SIGTERM);stop_time=time.time();a['stop_reason']='signal_or_budget_or_attempt_limit'
                if u>=b['max_gpu_hours']-.01 or (stop_time and time.time()-stop_time>90):os.killpg(child.pid,signal.SIGKILL)
                time.sleep(1)
            a['exit_code']=child.wait()
    except BaseException:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        a['terminal_status']='SUPERVISOR_EXCEPTION';raise
    finally:
        a.update(seconds=time.time()-start,ended_at=now(),log_sha256=sha(log) if log.exists() else None);write(LEDGER,b,replace=True)
    return a['exit_code']

def cpu_record(name,args):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    subprocess.run([sys.executable,'tools/record.py','--name',name,'--',sys.executable,*args],cwd=ROOT,env=env,check=True)

def main(mode):
    RUN.mkdir(parents=True,exist_ok=True)
    with (PARENT/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);b=load_ledger();frozen=contract()
        if mode=='engineering':
            if (RUN/'engineering/gpu_acceptance.json').exists():raise FileExistsError('Completed engineering must not repeat')
            code=allocate(b,'rr_rank_engineering_v1','engineering',.75)
            if code:raise SystemExit(code)
            cpu_record('rr_rank_diagnostic_engineering_postflight_v1',['code/rr_rank_v1/diagnostics.py','--postflight',str(RUN/'engineering/diagnostics')])
        acceptance=read(RUN/'engineering/gpu_acceptance.json')
        if acceptance['contract']!=frozen:raise ValueError('Stale engineering')
        if not (RUN/'engineering/diagnostics/postflight.json').exists():raise ValueError('Diagnostic index postflight incomplete')
        required=acceptance['conservative_training_gpu_hours']+acceptance['conservative_diagnostic_gpu_hours'];available=b['save_at_gpu_hours']-used(b)
        ready=required<=available
        admission=dict(status='ADMITTED' if ready else 'BUDGET_EXTENSION_REQUIRED',at=now(),cumulative_gpu_hours=used(b),remaining_to_hard_cap=b['max_gpu_hours']-used(b),
            required_training_reserve=acceptance['conservative_training_gpu_hours'],required_diagnostic_reserve=acceptance['conservative_diagnostic_gpu_hours'],available_before_save=available,
            minimum_cap_for_reservation=used(b)+required+.25,additional_cap_needed=max(0,required-available),ledger_sha256=sha(LEDGER),contract=frozen,
            training_status='NOT_RUN',diagnostic_status='ENGINEERING_ONLY; full144 final-weight tasks NOT_RUN')
        write(RUN/f'admission_{mode}.json',admission);print(json.dumps(admission,indent=2),flush=True)
        # Engineering mode never implicitly starts a long training job.
        if mode=='engineering' or not ready:return
        if (RUN/'training_manifest.json').exists():raise FileExistsError('Existing training requires explicit exact recovery')
        code=allocate(b,'rr_rank_train_full_v1','train',acceptance['conservative_training_gpu_hours'])
        if code:raise SystemExit(code)
        cpu_record('rr_rank_final_training_postflight_v1',['code/rr_rank_v1/postflight.py'])
        code=allocate(b,'rr_rank_final_diagnostics_v1','diagnostics',acceptance['conservative_diagnostic_gpu_hours'])
        if code:raise SystemExit(code)
        cpu_record('rr_rank_final_diagnostics_postflight_v1',['code/rr_rank_v1/diagnostics.py','--postflight',str(RUN/'diagnostics_final')])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['engineering','train'],required=True);main(p.parse_args().mode)
