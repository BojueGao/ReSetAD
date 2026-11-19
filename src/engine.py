"""One episode, one optimizer update; exact state restoration."""
import hashlib
import math
import random
from pathlib import Path
import numpy as np
import torch
from .models.model import Outputs
from .losses import criterion, context_loss, normality_targets


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def state_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(model):
    decay=[];other=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        # Linear/attention matrices only; tokens and spatial/monotone parameters excluded.
        (decay if p.ndim==2 and (name.endswith('.weight') or name.endswith('in_proj_weight')) else other).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':1e-4},{'params':other,'weight_decay':0.}],
                            lr=2e-4,betas=(.9,.999),eps=1e-8)


class StageScheduler:
    def __init__(self,optimizer,horizon):
        if horizon<=200:raise ValueError('Stage horizon must exceed linear warmup')
        self.optimizer=optimizer;self.horizon=horizon;self.steps=0
    def next_lr(self):
        step=self.steps+1
        if step<=200:return 2e-4*step/200
        return 2e-5+(2e-4-2e-5)*.5*(1+math.cos(math.pi*min(step-200,self.horizon-200)/(self.horizon-200)))
    def prepare(self):
        lr=self.next_lr()
        for group in self.optimizer.param_groups:group['lr']=lr
        return lr
    def advance(self):self.steps+=1
    def state_dict(self):return {'horizon':self.horizon,'steps':self.steps}
    def load_state_dict(self,state):
        if state['horizon']!=self.horizon:raise ValueError('Scheduler horizon changed')
        self.steps=state['steps']


def forward_episode(model,context,query,context_masks,query_masks,labels,query_chunk=4):
    state=model.prepare_context(context)
    outputs=[model.score_queries(state,query[i:i+query_chunk]) for i in range(0,len(query),query_chunk)]
    out=Outputs(torch.cat([o.logits for o in outputs]),torch.cat([o.image_logit for o in outputs]),
                state.normal_logits,torch.cat([o.contributions for o in outputs]),
                torch.cat([o.matches for o in outputs]),state)
    return criterion(out,context_masks,query_masks,labels),out


def train_episode(model,optimizer,scheduler,context,query,context_masks,query_masks,labels,query_chunk=4,phase='joint'):
    optimizer.zero_grad(set_to_none=True);lr=scheduler.prepare()
    if phase=='warmup':
        loss=context_loss(model.context_normality(context),normality_targets(context_masks))
        terms={'context':loss,'total':loss}
    else:terms,_=forward_episode(model,context,query,context_masks,query_masks,labels,query_chunk)
    if not all(torch.isfinite(v).all() for v in terms.values()):raise FloatingPointError('FAILED_NUMERICS: loss')
    terms['total'].backward()
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step();scheduler.advance()
    return {**{k:float(v.detach()) for k,v in terms.items()},'gradient_norm':float(grad),'lr':lr}


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path,model,optimizer,scheduler,sampler,metadata):
    path=Path(path)
    if path.exists():raise FileExistsError('Checkpoint retention: never overwrite an existing step')
    path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
               sampler=sampler.state_dict(),rng=rng_state(),metadata=metadata)
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def restore_checkpoint(path,model,optimizer,scheduler,sampler,expected):
    # Only our own project-produced checkpoints; contains NumPy/Python RNG structures.
    state=torch.load(path,map_location='cpu',weights_only=False)
    for key,value in expected.items():
        if state['metadata'].get(key)!=value:raise ValueError(f'Resume contract changed: {key}')
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler']);sampler.load_state_dict(state['sampler']);restore_rng(state['rng'])
    return state['metadata']
