"""Read-only live process identities, latest checkpoint hash and cumulative GPU budget."""
import sys,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import RUN,LEDGER,read,sha,now,contract
from budget import ticks,used,validate

def main():
    b=read(LEDGER);validate(b,read(RUN/'budget_prefix_snapshot.json')['ledger']);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    warnings=[];m=read(RUN/'manager_status.json') if (RUN/'manager_status.json').exists() else None
    def alive(r):return bool(r and r.get('pid') and r.get('start_ticks') and ticks(r['pid'])==r['start_ticks'] and r.get('boot_id',boot)==boot)
    jobpath=ROOT/'outputs/jobs/large518_pipeline_v1.json';job=read(jobpath) if jobpath.exists() else None
    if m and m['status']=='RUNNING' and not alive(m):warnings.append('Manager marked running but identity absent; do not relaunch')
    active=m.get('active') if m else None
    if active and not alive(active):warnings.append('Active child identity absent; check supervisor completion')
    if active and alive(active):
        stat=Path(f'/proc/{active["pid"]}/stat').read_text().split(') ',1)[1].split()
        if int(stat[1])!=m['pid']:warnings.append('Child parent mismatch')
    latest=read(RUN/'latest_checkpoint.json') if (RUN/'latest_checkpoint.json').exists() else None
    if latest and sha(latest['path'])!=latest['sha256']:warnings.append('Latest checkpoint SHA mismatch')
    progress=None
    logs=sorted(RUN.glob('episodes_from_*.jsonl'))
    if logs:
        with logs[-1].open('rb') as f:
            f.seek(max(0,logs[-1].stat().st_size-131072));lines=f.read().splitlines()
        for line in reversed(lines):
            try:progress=json.loads(line);break
            except (ValueError,UnicodeDecodeError):continue
    if job and job.get('ended_at'):
        if job['exit_code']!=0 or sha(ROOT/job['log'])!=job['log_sha256']:warnings.append('Root wrapper failed or log hash differs')
    value=used(b);initial=read(ROOT/'provenance/initialization.json')['old_used']
    print(json.dumps(dict(at=now(),boot_id=boot,manager_status=m,manager_alive=alive(m),root_job=job,root_alive=alive(job),
        active_alive=alive(active),latest_checkpoint=latest,last_training_log=progress,cumulative_gpu_hours=value,
        experiment_gpu_hours=value-initial,remaining_gpu_hours=b['max_gpu_hours']-value,ledger_sha256=sha(LEDGER),
        open_allocations=[a['name'] for a in b['attempts'] if a.get('seconds') is None],warnings=warnings),ensure_ascii=False,indent=2))
    if warnings:raise SystemExit(1)
if __name__=='__main__':main()
