"""Bind only the accepted final518 export to unchanged target manifests/evaluation."""
import sys,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import worker as w

def main():
    import yaml
    from target.cache import header
    assert not (w.RUN/'preflight.json').exists(),'Existing target binding'
    terminal=w.read(w.TRAIN/'postflight.json');diag=w.read(w.TRAIN/'diagnostics_final/postflight.json')
    assert terminal['status']=='READY_FOR_DELIVERY_REVIEW' and terminal['actual_updates']==30180
    assert diag['status']=='READY' and diag['tasks']==144 and diag['records']==576
    assert w.sha(w.TRAIN/'inference_model.pt')==terminal['export_sha256']
    for name,h in terminal['contract']['files'].items():assert w.sha(w.ROOT/name)==h,name
    b=w.read(w.LEDGER);assert all(a.get('seconds') is not None for a in b['attempts'])
    stats={};all_rows=[];files=dict(terminal['contract']['files'])
    for ds in w.DATASETS:
        folder=w.ROOT/'result/dataset_audit'/ds;a=w.read(folder/'audit.json')
        rows=w.records(ds);ann=w.read(folder/'manifest.json')
        assert a['status']=='ready' and not a['issues']
        assert w.sha(folder/'images.json')==a['images_sha256'] and w.sha(folder/'manifest.json')==a['manifest_sha256']
        assert rows==w.read(folder/'images.json')
        pairs={(r['path'],r['sha256']) for r in rows}|{(r['mask_path'],r['mask_sha256']) for r in ann if r['mask_path']}
        def verify(pair):
            path,h=pair;assert w.sha(path)==h,path
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(verify,pairs))
        plans={s:w.rrinfer.round_plan(rows,s) for s in w.SEEDS}
        for s,groups in plans.items():
            assert w.Counter(t['query'][0]['sample_id'] for g in groups for t in w.rotating_turns(g))==w.Counter(r['sample_id'] for r in rows)
        sample=max(plans[101],key=lambda g:(sum(r['height']*r['width'] for r in g),tuple(r['sample_id'] for r in g)))
        stats[ds]=dict(images=len(rows),raw_queries=3*len(rows),groups_per_round={str(s):len(g) for s,g in plans.items()},
                      benchmark_sample_ids=[r['sample_id'] for r in sample],native_pixels=sum(r['height']*r['width'] for r in rows))
        all_rows+=rows;print('TARGET_READY',ds,len(rows),flush=True)
    assert len(all_rows)==10666
    h=header(yaml.safe_load((w.ROOT/'configs/base.yaml').read_text()));assert h['shape']==[4,37,37,1024]
    unique=len({r['sha256'] for r in all_rows});pixels=sum(r['height']*r['width'] for r in all_rows)
    needed=unique*4*37*37*1024*4+pixels*17+100*2**30
    assert shutil.disk_usage(w.RUN).free>needed
    w.write(w.RUN/'preflight.json',dict(status='ready',files=files,datasets=stats,images=len(all_rows),raw_queries=3*len(all_rows),
        model_sha256=terminal['export_sha256'],model_state_sha256=terminal['final_model_state_hash'],cache_header=h,
        unique_byte_hashes=unique,required_space_bytes=needed,excluded=['realiad','ksdd'],source_only=['visa'],time=w.now()))
    print('FINAL_518_BOUND',terminal['export_sha256'],flush=True)
if __name__=='__main__':main()
