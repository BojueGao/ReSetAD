"""CPU-only migration of immutable inputs and authorized +96 allowance."""
import sys,json,shutil,time,socket,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import read,write,sha,now,RUN,PARENT,LEDGER
OLD=Path('/home/ubuntu/MyDisk/VSI-AD-Large')
TARGET=ROOT/'result/inference/large518_target12_v1'
DATASETS=('mvtec_ad','mpdd','btad','dtd_synthetic','brain_mri','br35h','headct','isic','cvc_clinicdb','cvc_colondb','tn3k','kvasir_seg')

def ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    except FileNotFoundError:return None
def cp(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        if sha(src)!=sha(dst):raise ValueError('Existing input differs: '+str(dst))
    else:
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    return sha(dst)

def main():
    if socket.gethostname()!='ubuntu-MZF2-AC0-00':raise RuntimeError('A100-15 only')
    if (ROOT/'provenance/initialization.json').exists():raise FileExistsError('Already initialized')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for name in ('large_source_pipeline_v3','large_target12_manager_v2'):
        j=read(OLD/'outputs/jobs'/f'{name}.json')
        assert j['exit_code']==0 and j['ended_at'] and ticks(j['pid'])!=j['start_ticks']
        assert sha(OLD/j['log'])==j['log_sha256']
    b=read(OLD/'result/rr_v1/gpu_budget.json')
    assert b['additional_cap_gpu_hours']==96 and b['max_gpu_hours']==b['prior_gpu_hours']+96
    assert all(a['seconds'] is not None and a['ended_at'] for a in b['attempts'])
    for a in b['attempts']:
        assert a.get('boot_id')!=boot or ticks(a.get('pid',0))!=a.get('start_ticks')
    initial_used=b['prior_gpu_hours']+sum(a['seconds']/3600 for a in b['attempts'])
    assert abs(initial_used-128.30185532232788)<1e-8
    # Do not manufacture new credit by deleting old attempts or changing their values.
    cp(OLD/'result/rr_v1/gpu_budget.json',ROOT/'provenance/large448_budget_snapshot.json')
    auth=dict(time=now(),source='User explicitly authorized +96 GPU hours for518 training and automatic inference',
              added_gpu_hours=96.,old_allowance=96.,new_allowance=192.,old_used=initial_used,
              old_ledger_path=str(OLD/'result/rr_v1/gpu_budget.json'),old_ledger_sha256=sha(OLD/'result/rr_v1/gpu_budget.json'),
              old_attempt_count=len(b['attempts']),old_prefix_gpu_hours=b['prior_gpu_hours'])
    write(ROOT/'provenance/budget_authorization.json',auth)
    b.update(additional_cap_gpu_hours=192.,max_gpu_hours=b['prior_gpu_hours']+192.,save_at_gpu_hours=b['prior_gpu_hours']+191.75,
             authorization='2026-10-28 +96 and2026-10-30 +96; prior history and all448attempts retained',
             extension_events=[auth])
    write(LEDGER,b);write(RUN/'budget_prefix_snapshot.json',dict(ledger=b,sha256=sha(LEDGER),total=initial_used,at=now()))
    inputs={}
    for rel in ('outputs/manifests/source_full.jsonl','result/rr_v1/source_neighbors.json','result/rr_rank_v1/training_plan.json','result/rr_rank_v1/diagnostic_manifest.json','configs/rr_rank_v1.json','provenance/backbone_weight.json'):
        inputs[rel]=cp(OLD/rel,ROOT/rel)
    # Reference the verified original pretrained backbone; do not redownload or modify.
    weight=read(ROOT/'provenance/backbone_weight.json')
    assert sha(OLD/'weights/dinov2_vitl14_pretrain.pth')==weight['sha256']
    old_target=OLD/'result/inference/large_target12_20261030_v1'
    for ds in DATASETS:
        for name in ('audit.json','manifest.json','images.json'):
            rel=f'result/dataset_audit/{ds}/{name}';inputs[rel]=cp(OLD/rel,ROOT/rel)
        rel=f'result/inference/large518_target12_v1/inputs/{ds}.json'
        inputs[rel]=cp(old_target/'inputs'/f'{ds}.json',ROOT/rel)
    rel='result/inference/large518_target12_v1/inputs/base_metrics_per_dataset.csv'
    inputs[rel]=cp(old_target/'metrics/metrics_per_dataset.csv',ROOT/rel)
    originals={}
    for folder in ('src','code/rr_rank_v1','code/target14_v1','code/large_v1','code/large_target_v2'):
        for p in (OLD/folder).rglob('*.py'):originals[str(p.relative_to(OLD))]=sha(p)
    unchanged={}
    for rel in ('code/rr_rank_v1/sampling.py','code/rr_rank_v1/inference.py','code/rr_rank_v1/diagnostics.py','code/rr_rank_v1/run.py','code/rr_rank_v1/postflight.py','src/engine.py','src/stream_metrics.py','code/target14_v1/grouping.py','code/target14_v1/common.py','code/target14_v1/audit.py','code/target14_v1/evaluate.py'):
        assert sha(ROOT/rel)==sha(OLD/rel),rel
        unchanged[rel]=sha(ROOT/rel)
    old_contract=read(OLD/'result/rr_rank_v1/postflight.json')['contract']
    write(ROOT/'provenance/frozen_inputs.json',dict(inputs=inputs,unchanged_science=unchanged,original_code=originals,
         original_root=str(OLD),original_contract_digest=old_contract['digest'],at=now()))
    assert sha(OLD/'result/rr_v1/gpu_budget.json')==auth['old_ledger_sha256']
    assert shutil.disk_usage(ROOT).free>1.2*1024**4
    write(ROOT/'provenance/initialization.json',dict(status='READY',originals_read_only=True,old_used=initial_used,
         available_gpu_hours=b['max_gpu_hours']-initial_used,old_attempt_count=len(b['attempts']),boot_id=boot,time=now()))
    print('INITIALIZED',initial_used,b['max_gpu_hours']-initial_used,flush=True)

if __name__=='__main__':main()
