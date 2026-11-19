"""Bounded nine-class source development, complete holdout selection, and exact resume."""
from collections import Counter, OrderedDict
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time
import traceback
import numpy as np
import torch
import yaml
from .baselines.fixed import FixedBaseline
from .cache import FeatureCache, cache_header, code_digest
from .data.adapters import source_mask
from .data.episodes import EpisodeSampler
from .data.manifest import load_rows, image_record
from .dev_protocol import CYCLE_STEPS, TOTAL_STEPS, effective_config, lock_run
from .engine import (seed_all, state_hash, optimizer_for, StageScheduler, train_episode,
                     save_checkpoint, restore_checkpoint, rng_state, restore_rng)
from .evidence import sha256, write_json, utc_now
from .gpu_budget import GPUBudget, BudgetStop
from .models.backbone import FrozenBackbone
from .models.model import VSIAD
from .runtime import source_gate, acceptance_gate, select_gpu
from .source_validation import predict_source, evaluate_source


def read(path):
    return json.loads(Path(path).read_text())


def run_development(args, cfg, parser):
    if cfg['execution']['run_until'] != 'DEV_LOCK':
        parser.error('Source development requires explicit DEV_LOCK scope')
    if args.seed != 17 or args.steps not in (None,TOTAL_STEPS):
        parser.error('Development fixes seed=17 and 22635 optimizer steps')
    if args.budget_gpu_hours not in (None,25):
        parser.error('Approved stage budget is 25 GPU hours')
    if args.dry_run:
        print(json.dumps(dict(mode='dev', run_until='DEV_LOCK', seed=17,
            warmup_steps=CYCLE_STEPS, joint_steps=4*CYCLE_STEPS, total_steps=TOTAL_STEPS,
            holdout=['pcb4','capsules','cashew'], group_seed=101, candidates=[1,2,3,4],
            maximum_gpu_hours=25, final_training=False, target_evaluation=False)))
        return
    args.run_id = args.run_id or 'dev_seed17_20261013_v1'
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('Unsafe run ID')
    train_rows, audit = source_gate(cfg)
    acceptance = acceptance_gate(cfg, args.config)
    holdout_rows = load_rows(Path(cfg['execution']['manifest_dir'])/'source_dev_holdout.jsonl')
    if len(train_rows) != 8414 or len(holdout_rows) != 2407:
        raise ValueError('Wrong source inventory')
    root = Path('outputs/runs')/args.run_id
    fresh = args.resume is None
    if fresh and root.exists():
        raise FileExistsError('Run already exists; inspect jobs and resume explicitly')
    if not fresh and not root.exists():
        raise FileNotFoundError('Resume run does not exist')
    root.mkdir(parents=True, exist_ok=True)
    guard = open('outputs/training.lock','a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code_hash, code_files = code_digest()
    header = cache_header(cfg)
    contract = dict(run_id=args.run_id, mode='dev', seed=17, protocol_version=cfg['protocol_version'],
        code_hash=code_hash, config_hash=sha256(args.config), data_hash=audit['manifest_sha256'],
        weights_hash=header['weight_sha'], cache_header=header)
    # Revalidate the exact audited bytes before new GPU work; never rebuild the split.
    for index,row in enumerate(train_rows+holdout_rows):
        if sha256(row['path']) != row['image_sha256']:
            raise ValueError('Source image changed after audit')
        if row['mask_path'] and sha256(row['mask_path']) != row['mask_sha256']:
            raise ValueError('Source mask changed after audit')
        if (index+1)%2000 == 0:
            print(f'source byte re-audit {index+1}/10821', flush=True)
    write_json(root/'source_reaudit.json', dict(status='READY', images=10821, masks=1200,
        manifest_sha256=audit['manifest_sha256'], source_only=True))
    gpu, inventory = select_gpu(cfg,args.gpu)
    budget = GPUBudget(root/'gpu_budget.json',25)
    stopped = []
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda s,f:stopped.append(s))
    def poll():
        budget.poll()
        if stopped:
            raise InterruptedError('Cooperative stop requested')
    seed_all(17); torch.set_num_threads(4)
    backbone = FrozenBackbone(cfg['execution']['backbone_repo'],cfg['execution']['backbone_weights']).cuda()
    model = VSIAD(args.node_chunk or cfg['runtime']['node_chunk']).cuda()
    if sum(p.numel() for p in model.parameters())!=713157:
        raise AssertionError('Joint parameter count changed')
    initial_hash = state_hash(model)
    sampler = EpisodeSampler(train_rows,17)
    stage = 'warmup'
    step = 0
    usage_c, usage_q = Counter(), Counter()
    if not fresh:
        checkpoint_data = torch.load(args.resume,map_location='cpu',weights_only=False)
        stage = checkpoint_data['metadata']['phase']
        if sha256(args.resume) != read(root/'latest_checkpoint.json')['sha256']:
            raise ValueError('Resume must use the latest verified checkpoint')
        del checkpoint_data
    model.set_stage(stage); model.train()
    optimizer = optimizer_for(model)
    scheduler = StageScheduler(optimizer,CYCLE_STEPS if stage=='warmup' else 4*CYCLE_STEPS)
    if fresh:
        sampler.begin_cycle('warmup')
    else:
        metadata = restore_checkpoint(args.resume,model,optimizer,scheduler,sampler,contract)
        step = metadata['step']; usage_c.update(metadata['usage_c']); usage_q.update(metadata['usage_q'])
        initial_hash = read(root/'run_manifest.json')['initial_model_hash']
    cache = FeatureCache('outputs/cache/source',header,backbone,'cuda',cfg['runtime']['encoder_batch_size'])
    holdout_cache = FeatureCache('outputs/cache/source_holdout',header,backbone,'cuda',cfg['runtime']['encoder_batch_size'])
    memory_features = OrderedDict()
    def cached_batch(rows, selected_cache=cache):
        values = []
        for row in rows:
            key = (str(selected_cache.root),row['sample_id'])
            if key not in memory_features:
                memory_features[key] = selected_cache.read(row)
            memory_features.move_to_end(key)
            values.append(memory_features[key])
            while len(memory_features)>256:
                memory_features.popitem(last=False)
        return torch.stack(values).cuda()
    by_id = {r['sample_id']:r for r in train_rows}
    @lru_cache(maxsize=512)
    def mask(sample_id):
        return source_mask(by_id[sample_id])
    hmap = {r['sample_id']:r for r in holdout_rows}
    records = [image_record(r) for r in holdout_rows]
    def validation_features(items):
        return cached_batch([hmap[r.sample_id] for r in items],holdout_cache)
    attempt = f'attempt_{step:06d}_{time.time_ns()}'
    log = (root/(attempt+'_episodes.jsonl')).open('x',encoding='utf-8',buffering=1)
    current_episode = None
    last_checkpoint_step = read(root/'latest_checkpoint.json')['step'] if not fresh else -1
    if fresh:
        manifest = {**contract, 'code_files':code_files, 'config_path':args.config,
            'source_audit_sha256':sha256(Path(cfg['execution']['manifest_dir'])/'source_audit.json'),
            'acceptance_sha256':sha256(cfg['execution']['acceptance_path']),
            'initial_model_hash':initial_hash, 'started_at':utc_now(), 'ended_at':None,
            'cwd':str(Path.cwd()), 'argv':vars(args), 'gpu':gpu, 'gpu_inventory':inventory,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                'cudnn':torch.backends.cudnn.version(),'dependency_freeze':subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True).splitlines()},
            'status':'NOT_RUN','orchestration_status':'PREPARING','target_access':False,'final_model_eligible':False}
        manifest['parameter_counts']={'new_modules':713157,'backbone':sum(p.numel() for p in backbone.parameters()),
                                      'warmup_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        write_json(root/'run_manifest.json',manifest)
        (root/'effective_config.yaml').write_text(yaml.safe_dump(effective_config(cfg),sort_keys=False),encoding='utf-8')
    else:
        manifest = read(root/'run_manifest.json')
    def update_state(operation, status='RUNNING'):
        latest = read(root/'latest_checkpoint.json') if (root/'latest_checkpoint.json').exists() else {}
        value = dict(project='VSI-AD',run_until='DEV_LOCK',stage=operation,run_id=args.run_id,
            orchestration_status=status,active_jobs=[] if status!='RUNNING' else [{'pid':os.getpid(),'attempt':attempt,'step':step}],
            checkpoint=latest.get('path'),checkpoint_sha256=latest.get('sha256'),checkpoint_step=latest.get('step'),
            remote_host='A100-15',remote_root=str(Path.cwd()),code_hash=code_hash,config_hash=contract['config_hash'],
            data_hash=contract['data_hash'],weights_hash=contract['weights_hash'],
            maximum_gpu_hours=25,consumed_gpu_hours=budget.consumed()/3600,
            experiments={'smoke':'READY','overfit':'READY','dev':'READY' if status=='COMPLETE' else 'NOT_RUN','final':'NOT_RUN','target':'NOT_RUN'},
            resume_entry=f'.venv/bin/python tools/train.py --config {args.config} --mode dev --seed 17 --gpu {args.gpu} --run-id {args.run_id} --resume {latest.get("path")} --budget-gpu-hours 25')
        write_json('outputs/execution_state.json',value)
        write_json(root/'progress.json',dict(step=step,total_steps=TOTAL_STEPS,phase=stage,cycle=sampler.cycle,
            operation=operation,orchestration_status=status,gpu_hours=budget.consumed()/3600,updated_at=utc_now()))
    def checkpoint():
        nonlocal last_checkpoint_step
        if last_checkpoint_step==step:
            return
        path = root/'checkpoints'/f'step_{step:06d}.pt'
        save_checkpoint(path,model,optimizer,scheduler,sampler,{**contract,'step':step,'phase':stage,
            'cycle':sampler.cycle,'usage_c':dict(usage_c),'usage_q':dict(usage_q),'attempt':attempt})
        write_json(root/'latest_checkpoint.json',dict(path=str(path),sha256=sha256(path),step=step,phase=stage,cycle=sampler.cycle))
        last_checkpoint_step=step
    def validate(model_to_use, directory, model_id, checkpoint_sha=None):
        saved_rng = rng_state()
        before = state_hash(model)
        try:
            update_state(f'SOURCE_VALIDATION_{model_id}')
            provenance = dict(run_id=args.run_id, model_id=model_id,train_seed=17 if model_id.startswith('joint') else None,
                              checkpoint_sha256=checkpoint_sha, code_hash=code_hash, data_hash=contract['data_hash'])
            predict_source(model_to_use,records,validation_features,directory,provenance,poll)
            return evaluate_source(directory,holdout_rows,calibrated=model_id.startswith('joint'),poll=poll)
        finally:
            restore_rng(saved_rng)
            model.set_stage(stage);model.train()
            if state_hash(model)!=before:
                raise AssertionError('Source validation changed training state')
    result_status, orchestration = 'NOT_RUN','RUNNING'
    torch.cuda.reset_peak_memory_stats()
    try:
        checkpoint();update_state('SOURCE_CACHE')
        for selected_cache,rows in ((cache,train_rows),(holdout_cache,holdout_rows)):
            for start in range(0,len(rows),32):
                poll();selected_cache.ensure(rows[start:start+32])
        # The two prescribed fixed rules are evaluated only on source holdout here.
        for kind in ('B0','B1'):
            if validate(FixedBaseline(kind,model.node_chunk).cuda(),root/'baselines'/kind,kind)['status']!='READY':
                raise ValueError('Incomplete source fixed-baseline validation')
        for cycle in range(5):
            planned_stage = 'warmup' if cycle==0 else 'joint'
            start, end = cycle*CYCLE_STEPS, (cycle+1)*CYCLE_STEPS
            if step<start:
                raise AssertionError('Development step schedule gap')
            if step<end:
                if stage!=planned_stage:
                    before=state_hash(model);stage=planned_stage;model.set_stage(stage)
                    optimizer=optimizer_for(model);scheduler=StageScheduler(optimizer,4*CYCLE_STEPS)
                    if state_hash(model)!=before or len(optimizer.state):
                        raise AssertionError('Stage transition must preserve values and reset momentum')
                    write_json(root/'stage_transition.json',dict(step=step,model_hash=before,optimizer_state_entries=0,
                        scheduler_horizon=4*CYCLE_STEPS,lr_first_joint_step=scheduler.next_lr()))
                if sampler.cycle!=cycle:
                    if step!=start:
                        raise AssertionError('Cannot reset sampler inside a cycle')
                    sampler.begin_cycle(stage)
                model.train()
                frozen = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
                while step<end:
                    poll();current_episode=sampler.next();e=current_episode;started=time.monotonic()
                    context=cached_batch(e.context)
                    cm=torch.stack([mask(r['sample_id']) for r in e.context]).cuda()
                    if stage=='joint':
                        query=cached_batch(e.query);qm=torch.stack([mask(r['sample_id']) for r in e.query]).cuda()
                        labels=torch.tensor([r['label'] for r in e.query],device='cuda')
                    else:
                        query=qm=labels=None
                    terms=train_episode(model,optimizer,scheduler,context,query,cm,qm,labels,
                                        query_chunk=args.query_chunk,phase=stage)
                    torch.cuda.synchronize();step+=1
                    usage_c.update(r['sample_id'] for r in e.context)
                    if stage=='joint':usage_q.update(r['sample_id'] for r in e.query)
                    entry=dict(step=step,optimizer_step_in_stage=scheduler.steps,phase=stage,cycle=cycle,category=e.category,
                        context_ids=[r['sample_id'] for r in e.context],query_ids=[r['sample_id'] for r in e.query],
                        NC=len(e.context),rho_image=e.ratio,rho_pixel=float(cm.mean()),**terms,
                        seconds=time.monotonic()-started,
                        context_use_counts={r['sample_id']:usage_c[r['sample_id']] for r in e.context},
                        query_use_counts={r['sample_id']:usage_q[r['sample_id']] for r in e.query})
                    log.write(json.dumps(entry)+'\n')
                    current_episode=None
                    del context,query,cm,qm,labels
                    if step%10==0 or step==1:
                        print(f'dev {stage} cycle={cycle} step={step}/{TOTAL_STEPS} loss={terms["total"]:.6f} sec={entry["seconds"]:.3f} gpu_hours={budget.consumed()/3600:.4f}',flush=True)
                        update_state(f'TRAIN_{stage.upper()}_{cycle}')
                    if step%100==0 or step==end:
                        checkpoint()
                for name,p in model.named_parameters():
                    if name in frozen and not torch.equal(p.detach().cpu(),frozen[name]):
                        raise AssertionError(f'Frozen warmup module changed: {name}')
                coverage = sampler.audit_coverage() if stage=='joint' else dict(steps=CYCLE_STEPS,query_supervision=False)
                coverage.update(cycle=cycle,phase=stage,query_counts=dict(sampler.query_counts),context_counts=dict(sampler.context_counts))
                write_json(root/f'coverage_{stage}_{cycle}.json',coverage)
            if cycle>0:
                coverage_path=root/f'coverage_joint_{cycle}.json'
                if not coverage_path.exists():
                    if step!=end or sampler.cycle!=cycle:
                        raise ValueError('Cannot recover coverage from a different cycle state')
                    coverage=sampler.audit_coverage()
                    coverage.update(cycle=cycle,phase='joint',query_counts=dict(sampler.query_counts),context_counts=dict(sampler.context_counts))
                    write_json(coverage_path,coverage)
                directory=root/'validation'/f'joint_{cycle}'
                if not (directory/'metrics.json').exists():
                    if step!=end:
                        raise ValueError('Missing validation for an earlier checkpoint; stop instead of using current weights')
                    if validate(model,directory,f'joint_{cycle}',sha256(root/'checkpoints'/f'step_{end:06d}.pt'))['status']!='READY':
                        raise ValueError('Incomplete holdout selection candidate')
        result_status,orchestration='READY','COMPLETE'
    except (BudgetStop,InterruptedError) as exc:
        orchestration='BUDGET_EXHAUSTED' if isinstance(exc,BudgetStop) else 'INTERRUPTED'
        # poll is only before an episode or during read-only validation: state is committed.
        checkpoint()
        write_json(root/(attempt+'_stop.json'),dict(reason=str(exc),step=step,time=utc_now()))
    except Exception as exc:
        orchestration='FAILED'
        write_json(root/(attempt+'_failure.json'),dict(exception=repr(exc),traceback=traceback.format_exc(),step=step,
            failed_episode=None if current_episode is None else dict(category=current_episode.category,
                context_ids=[r['sample_id'] for r in current_episode.context],query_ids=[r['sample_id'] for r in current_episode.query]),
            last_checkpoint=read(root/'latest_checkpoint.json')))
        raise
    finally:
        log.close();budget.finish()
        summary={**contract,'status':result_status,'orchestration_status':orchestration,'completed_steps':step,
            'optimizer_steps':step,'phase':stage,'cycle':sampler.cycle,'target_access':False,'final_model_eligible':False,
            'source_categories':sampler.categories,'gpu_hours':budget.consumed()/3600,'ended_at':utc_now(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'query_usage':dict(usage_q),'context_usage':dict(usage_c),'final_model_hash':state_hash(model)}
        write_json(root/'run_summary.json',summary)
        manifest.update(status=result_status,orchestration_status=orchestration,ended_at=summary['ended_at'])
        write_json(root/'run_manifest.json',manifest)
        update_state('DEV_LOCK' if orchestration=='COMPLETE' else stage,orchestration)
    if result_status=='READY':
        update_state('SOURCE_LOCK','LOCKING')
        try:
            locked=lock_run(root,root/'protocol_lock.json')
            lock_run(root,'outputs/protocol_lock.json')
            write_json(root/'lock_receipt.json',dict(status='READY',e_star=locked['e_star'],
                lock_sha256=sha256(root/'protocol_lock.json'),global_lock_sha256=sha256('outputs/protocol_lock.json')))
        except Exception as exc:
            update_state('SOURCE_LOCK','FAILED_LOCK')
            write_json(root/'lock_failure.json',dict(reason=repr(exc),time=utc_now()))
            raise
        update_state('DEV_LOCK','COMPLETE')
        print(json.dumps(dict(status='READY',run_id=args.run_id,e_star=locked['e_star'],J=locked['selection_J'],gpu_hours=budget.consumed()/3600)),flush=True)
    else:
        raise SystemExit(75 if orchestration=='BUDGET_EXHAUSTED' else 130)
