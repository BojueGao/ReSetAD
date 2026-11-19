"""Source-only epoch choice and immutable protocol locking."""
import copy
from collections import Counter
import json
import math
from pathlib import Path
import yaml
from .cache import canonical_hash, code_digest
from .evidence import sha256, write_json, utc_now

HOLDOUT = ('capsules', 'cashew', 'pcb4')
CYCLE_STEPS = 4527
TOTAL_STEPS = 22635


def choose_cycle(candidates, tolerance=1e-4):
    if sorted(candidates) != [1,2,3,4]:
        raise ValueError('All four joint candidates are mandatory')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in candidates.values()):
        raise ValueError('Invalid source J')
    maximum = max(candidates.values())
    # Compare to the global maximum, not a non-transitive sequential incumbent.
    return min(e for e,value in candidates.items() if maximum-value <= tolerance)


def effective_config(cfg):
    result = copy.deepcopy(cfg)
    result['implementation_contract'] = {
        'source_partition':'merged original VisA; 9 train / pcb4,capsules,cashew holdout',
        'source_development_seed':17, 'development_group_seed':101,
        'development_cycle_steps':4527, 'development_total_steps':22635,
        'warmup_trainable_prefixes':['encoder.embed.','encoder.isab1.','encoder.isab2.','normality.'],
        'warmup_loss':'L_ctx only, without joint 0.2 multiplier',
        'stage_transition':'preserve parameters; reconstruct optimizer and scheduler; no warmup momentum',
        'joint_scheduler_steps':18108, 'warmup_scheduler_steps':4527,
        'scheduler_step_index':'one-based for lr before each optimizer step',
        'scheduler_formula':'step<=200:2e-4*step/200; otherwise 2e-5+0.5*1.8e-4*(1+cos(pi*(step-200)/(horizon-200)))',
        'weight_decay_groups':{'linear_and_attention_matrices':1e-4,'bias_LN_tokens_spatial_monotone':0.0},
        'linear_initialization':'Xavier uniform; attention Q/K/V matrices separately; bias=0',
        'layernorm_initialization':{'weight':1.,'bias':0.},
        'tokens_initialization':{'distribution':'normal','mean':0.,'std':.02},
        'normality_last_bias':0., 'controller_last_weight':0., 'controller_first_layer':'Xavier uniform',
        'controller_initial_values':{'alpha_logits':0.,'omega_logits':0.,'lambda':.02,'tau':1.},
        'spatial_initial_values':{'center_logit':2.,'other_logits':0.,'gamma':.05},
        'monotone_initial_values':{'softplus_V':1.,'softplus_A':.25,'C':'8 equally spaced [-1,1]','beta0':-1.},
        'branch_order':'layer-major [2,5,8,11], scale-minor [1,3,5]',
        'anchor_rows_columns':list(range(1,32,2)), 'anchors_per_image':256,'query_patches':1024,
        'relation_axes':{'d_cc':['NC',256,12,'NC-1'],'d_qc':['NQ',1024,12,'NC']},
        'quantile_axis':'last reference-image axis','reliability_penalty':'inside anchor min',
        'jackknife':'average of quantile mixtures after deleting one reference image; exact sorted equivalent',
        'robust_stats':'median; MAD+0.01; v=m+3*s*tanh((x-m)/(3*s)); weighted mu and variance on v; sigma=sqrt(var+0.01^2)',
        'focal_probability_clamp':'only log arguments to [1e-6,1-1e-6]',
        'dice_scope':'anomalous queries only, un-clamped probabilities',
        'image_top_k':11,'image_score':'sigmoid(mean top-11 logits)',
        'pixel_score':'sigmoid patch logits then bilinear resize, align_corners=False',
        'source_image_transform':'EXIF transpose; RGB; PIL BICUBIC 448; ImageNet normalization',
        'source_mask_transform':'nonzero union; area resize then >0; assert positive support',
        'query_queues':'independent normal/anomaly queues; reset normal only at start of each joint cycle (S001)',
        'context_queues':'independent, persistent; exclude query ID/hash; no replacement within episode',
        'group_rng':'PCG64 full SHA256 integer of compact JSON [dataset_id,pool_id,seed]',
        'source_validation_dataset_id':'visa_source_holdout',
        'tail_rules':{'r0':'20 each','r12to19':'keep tail','r1to11':'merge into previous group',
                      'n3to11':'one group; Q=min(4,n-2)','n_below3':'NO_VALID_CONTEXT'},
        'query_rotation':'sequential at most 4; C=group minus complete current Q chunk; no query-to-query references',
        'context_exclusions':'same hash, known case; deduplicate hashes; no labels',
        'selection':'0.5*macro I-AUROC+0.5*macro AUPRO@0.30; earliest within 1e-4 of global maximum',
        'pro':'4096 globally equally spaced score thresholds; >=; 8-connectivity; equal regions; all background',
        'pro_curve':'empty endpoint; maximum PRO at duplicate FPR; linear interpolation to .30; normalized trapezoid',
        'pixel_ranking':'all original pixels; exact FP32 value ranks via integer radix counts, signed zero tied',
        'baseline_thresholds':'B0/B1 raw scores; F1/Dice/IoU threshold NA_NO_CALIBRATED_THRESHOLD',
        'missing_class_selection':'fail; never silently average fewer than all three holdout classes',
        'amp':False,'tf32':False,'deterministic_algorithms':True,
        'trainable_parameters_joint':713157,'frozen_backbone_parameters':86580480,
        'new_module_precision':'float32','backbone_extraction_precision':'float32','cache_precision':'float32 (E002)',
    }
    return result


def lock_run(run, output):
    run, output = Path(run), Path(output)
    manifest = json.loads((run/'run_manifest.json').read_text())
    summary = json.loads((run/'run_summary.json').read_text())
    if summary['completed_steps'] != TOTAL_STEPS or summary['status'] != 'READY':
        raise ValueError('Cannot lock incomplete development')
    if manifest['mode'] != 'dev' or manifest['seed'] != 17 or summary['target_access']:
        raise ValueError('Invalid source-only development provenance')
    current_code, _ = code_digest()
    if current_code != manifest['code_hash']:
        raise ValueError('Code changed after development')
    if sha256(manifest['config_path']) != manifest['config_hash']:
        raise ValueError('Configuration changed after development')
    config = yaml.safe_load((run/'effective_config.yaml').read_text())
    manifest_dir=Path(config['execution']['manifest_dir'])
    if sha256(manifest_dir/'source_full.jsonl')!=manifest['data_hash'] or sha256(manifest_dir/'source_audit.json')!=manifest['source_audit_sha256']:
        raise ValueError('Source inventory/audit changed after development')
    holdout=[json.loads(line) for line in (manifest_dir/'source_dev_holdout.jsonl').read_text().splitlines()]
    expected_ids=Counter(r['sample_id'] for r in holdout)
    if len(holdout)!=2407 or len(expected_ids)!=2407:
        raise ValueError('Wrong holdout inventory')
    evidence, candidates = {}, {}
    for cycle in range(1,5):
        root = run/'validation'/f'joint_{cycle}'
        metrics = json.loads((root/'metrics.json').read_text())
        seal = json.loads((root/'seal.json').read_text())
        coverage = json.loads((run/f'coverage_joint_{cycle}.json').read_text())
        if (metrics['status'] != 'READY' or set(metrics['pools']) != set(HOLDOUT)
                or metrics['group_seed'] != 101 or metrics['valid_classes'] != 3
                or not metrics['source_only'] or metrics['target_access']):
            raise ValueError('Source selection requires three valid complete classes')
        if (metrics['seal_sha256'] != sha256(root/'seal.json') or not seal['sealed']
                or seal['state_hash_before'] != seal['state_hash_after']
                or seal['inventory_sha256'] != sha256(root/'predictions.json')
                or seal['predicted_queries'] != 2407):
            raise ValueError('Invalid source prediction seal')
        predictions=json.loads((root/'predictions.json').read_text())
        if Counter(r['sample_id'] for r in predictions)!=expected_ids:
            raise ValueError('Each complete source holdout image must be predicted exactly once')
        if coverage['steps'] != CYCLE_STEPS or coverage['normal_min_count'] < 1:
            raise ValueError('Source joint coverage failed')
        iroc = sum(metrics['pools'][c]['image']['auroc'] for c in HOLDOUT)/3
        pro = sum(metrics['pools'][c]['aupro']['value'] for c in HOLDOUT)/3
        candidates[cycle] = .5*(iroc+pro)
        if abs(metrics['J']-candidates[cycle]) > 1e-12:
            raise ValueError('Stored J does not match class metrics')
        checkpoint = run/'checkpoints'/f'step_{(cycle+1)*CYCLE_STEPS:06d}.pt'
        if sha256(checkpoint) != seal['checkpoint_sha256']:
            raise ValueError('Validation/checkpoint mismatch')
        evidence[str(cycle)] = dict(metrics=metrics, metrics_sha256=sha256(root/'metrics.json'),
            seal_sha256=sha256(root/'seal.json'), coverage_sha256=sha256(run/f'coverage_joint_{cycle}.json'),
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    best = choose_cycle(candidates)
    config['train']['final_joint_cycles'] = best
    config['locked_final_schedule'] = {'warmup_steps':6036, 'joint_steps':6036*best,
                                     'joint_scheduler_horizon_steps':4*6036,'total_steps':6036*(1+best)}
    resolved_path=output.parent/('locked_effective_config.yaml' if output.parent==run else 'effective_config.yaml')
    resolved_text=yaml.safe_dump(config,sort_keys=False)
    if resolved_path.exists():
        if resolved_path.read_text()!=resolved_text:
            raise ValueError('Resolved effective configuration is immutable')
    else:
        resolved_path.parent.mkdir(parents=True,exist_ok=True)
        resolved_path.write_text(resolved_text,encoding='utf-8')
    value = dict(status='READY', scope='DEV_LOCK', protocol_version=manifest['protocol_version'],
        run_id=manifest['run_id'], source_development_seed=17, development_group_seed=101,
        best_joint_cycle=best, e_star=best, tie_tolerance=1e-4, selection_rule='earliest within tolerance of global maximum',
        source_candidates=evidence, selection_J={str(k):v for k,v in candidates.items()},
        effective_config=config, effective_config_sha256=canonical_hash(config),
        effective_config_file=str(resolved_path),effective_config_file_sha256=sha256(resolved_path),
        code_hash=manifest['code_hash'], code_files=manifest['code_files'], config_hash=manifest['config_hash'],
        data_hash=manifest['data_hash'], weights_hash=manifest['weights_hash'], environment=manifest['environment'],
        formula_spec_sha256=sha256('specs/experiment_spec.md'), source_audit_sha256=manifest['source_audit_sha256'],
        selected_development_checkpoint=evidence[str(best)]['checkpoint'],
        selected_development_checkpoint_sha256=evidence[str(best)]['checkpoint_sha256'],
        final_training_seeds=[17,42,3407], final_group_seeds=[101,202,303],
        final_model_eligible=False, final_requires_fresh_initialization=True,
        target_results_used=False, final_training_status='NOT_RUN', target_evaluation_status='NOT_RUN')
    if output.exists():
        previous = json.loads(output.read_text())
        old = {k:v for k,v in previous.items() if k != 'locked_at'}
        if old != value:
            raise ValueError('Immutable protocol lock already exists with other contents')
        return previous
    value['locked_at'] = utc_now()
    write_json(output, value)
    return value
