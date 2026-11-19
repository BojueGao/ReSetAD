import argparse
import json
import os
import subprocess
from pathlib import Path
import yaml
from .evidence import sha256
from .data.manifest import load_rows
from .data.adapters import HOLDOUT


def config_load(path):
    with open(path,encoding='utf-8') as f:cfg=yaml.safe_load(f)
    if cfg['protocol_version']!='vsi_ad_v1.0.1_s001':raise ValueError('Unknown protocol version')
    checks=[(cfg['preprocess']['size'],[448,448]),(cfg['model']['layer_indices_zero_based'],[2,5,8,11]),
            (cfg['model']['neighborhood_sizes'],[1,3,5]),(cfg['sampler']['query_size'],4),
            (cfg['model']['backbone'],'dinov2_vitb14')]
    if any(a!=b for a,b in checks):raise ValueError('Core scientific configuration changed')
    return cfg


def source_gate(cfg):
    root=Path(cfg['execution']['manifest_dir']);audit=json.loads((root/'source_audit.json').read_text())
    if audit['status']!='READY' or sha256(root/'source_full.jsonl')!=audit['manifest_sha256']:raise ValueError('Source audit not READY or inventory changed')
    allrows=load_rows(root/'source_full.jsonl');rows=load_rows(root/'source_dev_train.jsonl');holdout=load_rows(root/'source_dev_holdout.jsonl')
    expected=[r for r in allrows if r['category_id'] not in HOLDOUT]
    if rows!=expected or holdout!=[r for r in allrows if r['category_id'] in HOLDOUT]:raise ValueError('Source partition changed')
    if len({r['category_id'] for r in rows})!=9:raise ValueError('Need 9 development training classes')
    if {r['image_sha256'] for r in rows}&{r['image_sha256'] for r in holdout}:raise ValueError('Source holdout leakage')
    return rows,audit


def select_gpu(cfg,index):
    if index not in cfg['execution']['approved_gpus']:raise PermissionError('GPU not authorized')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    inventory=[]
    for line in text.strip().splitlines():
        i,uuid,name,used,total=[v.strip() for v in line.split(',')]
        inventory.append(dict(index=int(i),uuid=uuid,name=name,used_mib=int(used),total_mib=int(total)))
    gpu=next(g for g in inventory if g['index']==index)
    if gpu['used_mib']>=cfg['execution']['prelaunch_memory_used_max_gib_exclusive']*1024:raise PermissionError('GPU exceeds authorized prelaunch occupancy threshold')
    os.environ['CUDA_VISIBLE_DEVICES']=str(index);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    return gpu,inventory


def acceptance_gate(cfg,config_path='configs/base.yaml'):
    path=Path(cfg['execution'].get('acceptance_path','outputs/acceptance/acceptance.json'))
    if not path.exists():raise ValueError('Section-39 acceptance not run')
    data=json.loads(path.read_text())
    if data['status']!='READY':raise ValueError('Section-39 acceptance failed')
    # The trained numerical implementation must be the tested implementation.
    for name,value in data['tested_code'].items():
        if sha256(name)!=value:raise ValueError(f'Acceptance stale: {name}')
    if data['config_sha256']!=sha256(config_path):raise ValueError('Acceptance configuration stale')
    return data
