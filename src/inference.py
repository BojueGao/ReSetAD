"""Label-free inference and sealed raw predictions. No optimizer or training API."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from .data.grouping import groups,rounds
from .engine import state_hash
from .models.scorer import pixel_maps
from .evidence import write_json,sha256


@torch.inference_mode()
def predict_inventory(model,records,feature_provider,dataset_id,seed=101):
    model.eval();before=state_hash(model);predictions=[];failures=[]
    for pool in sorted({r.pool_id for r in records}):
        pool_records=[r for r in records if r.pool_id==pool]
        for group_index,group in enumerate(groups(pool_records,dataset_id,pool,seed)):
            for turn in rounds(group):
                if turn['status']!='READY':
                    failures.extend({'sample_id':r.sample_id,'status':'NO_VALID_CONTEXT'} for r in turn['query']);continue
                context=turn['context'];query=turn['query']
                state=model.prepare_context(feature_provider(context))
                out=model.score_queries(state,feature_provider(query))
                maps=pixel_maps(out.logits,[(r.height,r.width) for r in query]);scores=out.image_logit.sigmoid()
                for row,score,map_ in zip(query,scores,maps):
                    if not torch.isfinite(score) or not torch.isfinite(map_).all():raise FloatingPointError('FAILED_NUMERICS')
                    predictions.append(dict(sample_id=row.sample_id,pool_id=pool,image_score=float(score),
                                            pixel_map=map_.cpu().numpy().astype(np.float32),group=group_index,
                                            context_ids=[r.sample_id for r in context],query_ids=[r.sample_id for r in query]))
    after=state_hash(model)
    if before!=after:raise AssertionError('Target model state changed')
    expected=Counter(r.sample_id for r in records);actual=Counter(r['sample_id'] for r in predictions)
    complete=not failures and expected==actual
    return predictions,dict(status='READY' if complete else 'INCOMPLETE',expected_queries=len(records),predicted_queries=len(predictions),
                            failures=failures,state_hash_before=before,state_hash_after=after,group_seed=seed)


def seal_predictions(directory,predictions,report,provenance):
    root=Path(directory)
    if root.exists():raise FileExistsError('Sealed prediction directory is immutable')
    root.mkdir(parents=True);records=[]
    for index,pred in enumerate(predictions):
        path=root/f'{index:07d}.npz';np.savez_compressed(path,pixel_map=pred['pixel_map'].astype(np.float32))
        records.append({**{k:v for k,v in pred.items() if k!='pixel_map'},'map_file':path.name,'map_sha256':sha256(path)})
    write_json(root/'predictions.json',records)
    seal={**report,**provenance,'inventory_sha256':sha256(root/'predictions.json'),'sealed':True}
    write_json(root/'seal.json',seal)
    return seal
