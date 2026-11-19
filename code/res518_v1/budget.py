"""Additional96-hour ledger, immutable history, A100-only process envelopes."""
import sys,time,socket,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import RUN,PARENT,LEDGER,read,sha

def ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    except FileNotFoundError:return None

def used(b,epoch=None):
    epoch=time.time() if epoch is None else epoch
    return b['prior_gpu_hours']+sum(a['seconds'] if a.get('seconds') is not None else max(0,epoch-a['start_epoch']) for a in b['attempts'])/3600

def validate(b,prefix):
    if b['prior_gpu_hours']!=prefix['prior_gpu_hours'] or b['attempts'][:len(prefix['attempts'])]!=prefix['attempts']:
        raise ValueError('Historical ledger prefix changed')
    if b.get('additional_cap_gpu_hours')!=192. or b['max_gpu_hours']!=b['prior_gpu_hours']+192 or b['save_at_gpu_hours']!=b['prior_gpu_hours']+191.75:
        raise ValueError('Additional budget authorization changed')
    if b['prior_ledger']!=prefix['prior_ledger'] or b['prior_ledger_sha256']!=prefix['prior_ledger_sha256']:
        raise ValueError('Historical ledger reference changed')

def load():
    b=read(LEDGER);prefix=read(RUN/'budget_prefix_snapshot.json')['ledger'];validate(b,prefix)
    if sha(b['prior_ledger'])!=b['prior_ledger_sha256']:raise ValueError('History snapshot changed')
    if any(a.get('seconds') is None for a in b['attempts']):raise RuntimeError('Open allocation: audit original PID/start_ticks/boot_id before recovery')
    return b

def free_device():
    if socket.gethostname()!='ubuntu-MZF2-AC0-00':raise RuntimeError('A100-15 only')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(rows)!=4 or any('A100-SXM4-40GB' not in r or int(r.split(',')[2])!=40960 for r in rows):raise RuntimeError('Wrong GPU host')
    free=[int(r.split(',')[0]) for r in rows if int(r.rsplit(',',1)[1])<100]
    if not free:raise RuntimeError('No exclusive free A100; other tasks untouched')
    return rows,free[0]

def admission(b,required):
    if not 0<required<float('inf'):raise ValueError('Finite positive full-duration estimate required')
    return required<=b['save_at_gpu_hours']-used(b)
