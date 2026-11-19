"""Complete source-holdout inference, immutable round shards, and post-seal evaluation."""
from collections import Counter
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from .data.grouping import groups, rounds
from .engine import state_hash
from .evidence import sha256, write_json, utc_now
from .cache import canonical_hash
from .metrics import ranking, localization
from .models.scorer import pixel_maps
from .stream_metrics import ExactPixelRanks, StreamingPRO


def read(path):
    return json.loads(Path(path).read_text())


def validate_seal(root):
    root = Path(root)
    seal = read(root/'seal.json')
    if not seal.get('sealed') or seal['status'] != 'READY':
        raise ValueError('Source predictions must be sealed and complete before label association')
    if sha256(root/'predictions.json') != seal['inventory_sha256']:
        raise ValueError('Prediction inventory changed after sealing')
    return seal, read(root/'predictions.json')


@torch.inference_mode()
def predict_source(model, records, features, root, provenance, poll=lambda: None):
    """ImageRecord-only; no label/mask table accepted by the inference boundary."""
    root = Path(root)
    model.eval()
    before = state_hash(model)
    contract = {**provenance, 'dataset_id':'visa_source_holdout', 'group_seed':101,
                'model_state_sha256':before,
                'records_sha256':canonical_hash([vars(r) for r in records])}
    if (root/'contract.json').exists():
        if read(root/'contract.json') != contract:
            raise ValueError('Cannot resume a prediction directory with different inputs/model')
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root/'contract.json', contract)
    if (root/'seal.json').exists():
        return validate_seal(root)[0]
    (root/'maps').mkdir(exist_ok=True)
    (root/'rounds').mkdir(exist_ok=True)
    predictions, grouping = [], []
    baseline = getattr(model, 'kind', None)
    round_count = 0
    for pool in sorted({r.pool_id for r in records}):
        pool_rows = [r for r in records if r.pool_id == pool]
        for group_id, group in enumerate(groups(pool_rows, 'visa_source_holdout', pool, 101)):
            for turn_index, turn in enumerate(rounds(group)):
                poll()
                if turn['status'] != 'READY':
                    raise ValueError('NO_VALID_CONTEXT in full source holdout')
                c, q = turn['context'], turn['query']
                description = dict(dataset='visa_source_holdout', pool=pool, group_seed=101,
                                   group_id=group_id, round=turn_index,
                                   query_ids=[r.sample_id for r in q], context_ids=[r.sample_id for r in c],
                                   actual_context_size=len(c), filter_rule='same hash or known case; no labels',
                                   excluded_candidate_ids=[r.sample_id for r in group if r not in c and r not in q])
                shard_path = root/'rounds'/f'{pool}_{group_id:04d}_{turn_index:02d}.json'
                if shard_path.exists():
                    shard = read(shard_path)
                    if shard['grouping'] != description:
                        raise ValueError('Grouping changed on prediction resume')
                    for row in shard['predictions']:
                        if sha256(root/row['map_file']) != row['map_sha256']:
                            raise ValueError('Previously completed prediction map changed')
                else:
                    state = model.prepare_context(features(c))
                    out = model.score_queries(state, features(q))
                    if baseline:
                        maps = [F.interpolate(v[None,None], size=(r.height,r.width), mode='bilinear',
                                              align_corners=False)[0,0] for r,v in zip(q,out['raw_map'])]
                        scores, logits = out['raw_image_score'], [None]*len(q)
                    else:
                        maps = pixel_maps(out.logits, [(r.height,r.width) for r in q])
                        scores, logits = out.image_logit.sigmoid(), out.image_logit
                    entries = []
                    for record, score, logit, value in zip(q, scores, logits, maps):
                        array = value.cpu().numpy().astype(np.float32)
                        if not np.isfinite(array).all() or not np.isfinite(float(score)):
                            raise FloatingPointError('FAILED_NUMERICS in source validation')
                        destination = root/'maps'/f'{record.sample_id}.npz'
                        if destination.exists():
                            # A crash before the round commit can leave an already durable map.
                            if not np.array_equal(np.load(destination)['pixel_map'], array):
                                raise ValueError('Uncommitted map differs on exact prediction resume')
                        else:
                            temporary = destination.with_suffix('.tmp')
                            with temporary.open('wb') as f:
                                np.savez_compressed(f, pixel_map=array)
                            temporary.replace(destination)
                        entries.append(dict(sample_id=record.sample_id, pool_id=pool,
                            image_score=float(score), image_logit=None if logit is None else float(logit),
                            map_file=destination.relative_to(root).as_posix(), map_sha256=sha256(destination),
                            height=record.height, width=record.width, pixel_min=float(array.min()),
                            pixel_max=float(array.max()), context_hash=canonical_hash(description['context_ids']),
                            group=group_id, round=turn_index, **provenance))
                    shard = {'grouping':description, 'predictions':entries}
                    write_json(shard_path, shard)
                    del state, out, maps
                predictions.extend(shard['predictions'])
                grouping.append(description)
                round_count += 1
                if round_count % 10 == 0:
                    print(f'source inference {provenance["model_id"]} {len(predictions)}/{len(records)}', flush=True)
                    write_json(root/'progress.json', dict(predicted_queries=len(predictions), expected_queries=len(records), time=utc_now()))
    if Counter(r.sample_id for r in records) != Counter(r['sample_id'] for r in predictions):
        raise ValueError('INCOMPLETE: source query coverage mismatch')
    after = state_hash(model)
    if before != after:
        raise AssertionError('Source validation changed model parameters/buffers')
    write_json(root/'predictions.json', predictions)
    with (root/'grouping.jsonl').open('w') as f:
        for row in grouping:
            f.write(json.dumps(row)+'\n')
    with (root/'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(predictions)
    seal = dict(**contract, status='READY', sealed=True, sealed_at=utc_now(),
                expected_queries=len(records), predicted_queries=len(predictions),
                state_hash_before=before, state_hash_after=after,
                inventory_sha256=sha256(root/'predictions.json'), grouping_sha256=sha256(root/'grouping.jsonl'),
                source_only=True, target_access=False)
    write_json(root/'seal.json', seal)
    return seal


def evaluate_source(root, holdout_rows, calibrated=True, poll=lambda: None):
    root = Path(root)
    seal, predictions = validate_seal(root)
    if not seal.get('source_only') or seal.get('target_access'):
        raise ValueError('Only source holdout is authorized')
    if (root/'metrics.json').exists():
        result = read(root/'metrics.json')
        if result['seal_sha256'] != sha256(root/'seal.json'):
            raise ValueError('Metrics belong to another seal')
        return result
    # Label/mask lookup is constructed only after the complete prediction seal above.
    annotation = {r['sample_id']:r for r in holdout_rows}
    if len(annotation) != len(holdout_rows) or set(annotation) != {r['sample_id'] for r in predictions}:
        raise ValueError('Source evaluation inventory mismatch')
    results = {}
    scratch = root/'metric_scratch'/f'attempt_{time.time_ns()}'
    scratch.mkdir(parents=True)
    write_json(scratch/'registry.json', dict(purpose='reproducible exact metric scratch',
        logical_sparse_bytes_per_class=2*2**32*8, retained=True, not_raw_predictions=True))
    for pool in sorted({r['pool_id'] for r in predictions}):
        items = [r for r in predictions if r['pool_id'] == pool]
        ranker = ExactPixelRanks(scratch/pool)
        pro = StreamingPRO(min(r['pixel_min'] for r in items), max(r['pixel_max'] for r in items))
        ys, scores, loc = [], [], []
        started = time.monotonic()
        try:
            for index, row in enumerate(items):
                poll()
                info = annotation[row['sample_id']]
                ys.append(info['label']); scores.append(row['image_score'])
                if info['category_id'] != pool or sha256(root/row['map_file']) != row['map_sha256']:
                    raise ValueError('Source map/class integrity failure')
                with np.load(root/row['map_file']) as data:
                    score = data['pixel_map']
                if score.dtype != np.float32 or score.shape != (row['height'],row['width']):
                    raise ValueError('Original-resolution FP32 maps required')
                if info['label'] == 0:
                    mask = np.zeros(score.shape, dtype=bool)
                else:
                    if sha256(info['mask_path']) != info['mask_sha256']:
                        raise ValueError('Holdout mask bytes changed')
                    with Image.open(info['mask_path']) as f:
                        mask = np.asarray(ImageOps.exif_transpose(f)) != 0
                ranker.add(mask, score); pro.add(mask, score)
                if calibrated and info['label'] == 1:
                    loc.append(localization(mask, score))
                if (index+1) % 50 == 0:
                    print(f'source metrics {pool} {index+1}/{len(items)} elapsed={time.monotonic()-started:.1f}s', flush=True)
            results[pool] = dict(image=ranking(ys,scores), pixel=ranker.result(poll), aupro=pro.result(),
                samples=len(ys), positive_images=int(sum(ys)), negative_images=len(ys)-int(sum(ys)),
                image_f1_050=float(f1_score(ys,np.asarray(scores)>=.5,zero_division=0)) if calibrated else None,
                dice=float(np.mean([x['dice'] for x in loc])) if loc else None,
                iou=float(np.mean([x['iou'] for x in loc])) if loc else None,
                threshold_status='READY' if calibrated else 'NA_NO_CALIBRATED_THRESHOLD',
                seconds=time.monotonic()-started, score_direction='higher_is_more_anomalous')
            write_json(root/f'metrics_{pool}.json', results[pool])
        finally:
            ranker.close()
    valid = [r for r in results.values() if r['image']['status']=='READY' and r['aupro']['status']=='READY']
    result = dict(status='READY' if len(valid)==3 and len(results)==3 else 'INCOMPLETE',
        pools=results, valid_classes=len(valid), total_classes=len(results), group_seed=101,
        seal_sha256=sha256(root/'seal.json'), source_only=True, target_access=False,
        macro_i_auroc=float(np.mean([r['image']['auroc'] for r in valid])) if valid else None,
        macro_aupro=float(np.mean([r['aupro']['value'] for r in valid])) if valid else None)
    result['J'] = .5*(result['macro_i_auroc']+result['macro_aupro']) if result['status']=='READY' else None
    write_json(root/'metrics.json', result)
    with (root/'metrics_per_class.csv').open('w',newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['pool','I_AUROC','I_AP','P_AUROC','P_AP','AUPRO_030','N','positive_images','GT_regions','valid_pixels','status'])
        for pool,row in results.items():
            writer.writerow([pool,row['image']['auroc'],row['image']['ap'],row['pixel']['auroc'],row['pixel']['ap'],
                row['aupro']['value'],row['samples'],row['positive_images'],row['aupro'].get('regions'),row['pixel']['valid_pixels'],
                'READY' if row['image']['status']==row['pixel']['status']==row['aupro']['status']=='READY' else 'INCOMPLETE'])
    return result
