"""Compatible label-free whole-set forward API; no evaluation or launcher."""
from common import ROOT
import importlib.util
import torch
from model import fuse_maps
from src.engine import state_hash
_spec=importlib.util.spec_from_file_location('rr_existing_balanced_groups',ROOT/'code/target14_v1/grouping.py')
_grouping=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_grouping)
balanced_groups=_grouping.balanced_groups
rotating_turns=_grouping.rotating_turns


@torch.inference_mode()
def predict_group(model,group,feature_provider):
    if any(any(k in r for k in ('label','image_label','mask_path','anomaly','y')) for r in group):raise ValueError('Labels forbidden in RR inference')
    before=state_hash(model);previous_mode=model.training;model.eval();outputs={};contexts={}
    try:
        for turn in rotating_turns(group):
            key=tuple(r['sample_id'] for r in turn['context'])
            if key not in contexts:contexts[key]=model.prepare_context(feature_provider(turn['context']))
            row=turn['query'][0]
            out=model.score_queries(contexts[key],feature_provider([row]),[(row['height'],row['width'])])
            if row['sample_id'] in outputs:raise AssertionError('Duplicate query evaluation')
            outputs[row['sample_id']]=out['maps'][0].cpu()
        if set(outputs)!={r['sample_id'] for r in group}:raise AssertionError('Coverage mismatch')
        if before!=state_hash(model):raise AssertionError('Inference changed model parameters/state')
        return outputs
    finally:model.train(previous_mode)


def round_plan(records,seed):
    return [g for pool in sorted({r['pool_id'] for r in records}) for g in balanced_groups([r for r in records if r['pool_id']==pool],seed)]
