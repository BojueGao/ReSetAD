"""Fixed source pipeline: CPU gates, metered cache/engineering/train/diagnostics."""
import sys,os,time,signal,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import RUN,PARENT,LEDGER,read,write,sha,now,contract
from budget import ticks,used,load,free_device,admission

class Pipeline:
    def __init__(self):
        self.b=load();self.expected=read(LEDGER);self.active=None;self.done=[];self.stage='starting';self.stopping=False
        self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.frozen=contract()
        if (RUN/'manager_status.json').exists():raise FileExistsError('Existing manager evidence; no implicit relaunch')
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda sig,frame:setattr(self,'stopping',True))
    def persist(self):
        if read(LEDGER)!=self.expected:raise RuntimeError('Concurrent ledger writer detected')
        write(LEDGER,self.b,replace=True);self.expected=read(LEDGER)
    def status(self,state='RUNNING',error=None):
        write(RUN/'manager_status.json',dict(status=state,stage=self.stage,pid=os.getpid(),start_ticks=ticks(os.getpid()),
              boot_id=self.boot,active=self.active,completed=self.done,cumulative_gpu_hours=used(self.b),
              new_gpu_hours=used(self.b)-self.b['prior_gpu_hours'],new_remaining_gpu_hours=self.b['max_gpu_hours']-used(self.b),
              error=error,at=now()),replace=True)
    def step(self,name,args,hours=None):
        if self.stopping:raise RuntimeError('Pipeline stop requested')
        if contract()!=self.frozen:raise ValueError('Active scientific/engineering contract changed')
        self.stage=name;path=RUN/'jobs'/(name+'.json');log=RUN/'jobs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() or log.exists():raise FileExistsError('Attempt exists; explicit recovery required')
        gpu=None;devices=None
        if hours is not None:
            if not admission(self.b,hours):raise RuntimeError('Full allocation does not fit new allowance')
            devices,gpu=free_device()
        argv=[sys.executable,'-u',*args]
        r=dict(name=name,run='dinov2_large518_v1',command=argv,gpu=gpu,device_snapshot=devices,start_epoch=time.time(),
               started_at=now(),seconds=None,ended_at=None,exit_code=None,boot_id=self.boot,log=str(log),limit_hours=hours)
        if gpu is not None:self.b['attempts'].append(r);self.persist()
        write(path,r);self.active=r;self.status();child=None;stop_at=None
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'' if gpu is None else str(gpu),'OMP_NUM_THREADS':'1' if gpu is None else '4',
             'MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        try:
            with log.open('xb',buffering=0) as output:
                child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                r.update(pid=child.pid,start_ticks=ticks(child.pid),ppid=os.getpid())
                if gpu is not None:self.persist()
                write(path,r,replace=True);self.status();last=0
                while child.poll() is None:
                    elapsed=time.time()-r['start_epoch']
                    stop=self.stopping or (gpu is not None and (used(self.b)>=self.b['save_at_gpu_hours'] or elapsed>=hours*3600))
                    if stop and stop_at is None:
                        stop_at=time.time();r['stop_reason']='signal_or_budget_or_allocation_limit';os.killpg(child.pid,signal.SIGTERM)
                    if stop_at and (time.time()-stop_at>90 or (gpu is not None and used(self.b)>=self.b['max_gpu_hours']-.01)):
                        os.killpg(child.pid,signal.SIGKILL)
                    if time.time()-last>10:self.status();last=time.time()
                    time.sleep(1)
                r['exit_code']=child.wait()
        except BaseException:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            if child is not None:r['exit_code']=child.returncode
            r['terminal_status']='SUPERVISOR_EXCEPTION';raise
        finally:
            r.update(seconds=time.time()-r['start_epoch'],ended_at=now(),log_sha256=sha(log) if log.exists() else None)
            if gpu is not None:self.persist()
            write(path,r,replace=True);self.done.append(r);self.active=None;self.status()
        print('STEP_EXIT',name,r['exit_code'],'new_GPUh',used(self.b)-self.b['prior_gpu_hours'],flush=True)
        if r['exit_code']!=0:raise RuntimeError(f'{name} failed; evidence retained; no automatic retry')

def main():
    os.chdir(ROOT);manager=None
    with (PARENT/'large_manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:manager=Pipeline();manager.run()
        except BaseException as error:
            write(RUN/f'failure_{time.time_ns()}.json',dict(error=repr(error),traceback=traceback.format_exc(),at=now()))
            if manager is not None:manager.status('FAILED',repr(error))
            raise
if __name__=='__main__':main()
