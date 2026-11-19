"""Only the image-supervision block changes; pixel/ref/inference remain RR."""
import math
import torch
import torch.nn.functional as F
from src.losses import context_loss,normality_targets
from model import image_score

def pixel_terms(logits,masks,labels):
    probs=F.interpolate(logits.sigmoid()[:,None],size=tuple(masks.shape[-2:]),mode='bilinear',align_corners=False)[:,0]
    p=probs.clamp(1e-6,1-1e-6)
    focal=(-.75*masks*(1-probs).square()*p.log()-.25*(1-masks)*probs.square()*torch.log1p(-p)).mean()
    positive=labels==1
    if positive.any():
        if not (masks[positive].flatten(1).sum(1)>0).all():raise ValueError('Empty anomaly query mask')
        pred=probs[positive].flatten(1);gt=masks[positive].flatten(1)
        dice=(1-(2*(pred*gt).sum(1)+1)/(pred.sum(1)+gt.sum(1)+1)).mean()
    else:dice=probs.sum()*0
    return dict(focal=focal,dice=dice,pixel=focal+.5*dice)

def evidence_from_distance(distance,sizes,rho,bias):
    a=F.softplus(rho).detach();b=bias.detach()
    if not torch.isfinite(a) or a<=1e-6:raise FloatingPointError('Rank slope numerical guard would alter equation')
    p=(a*distance+b).sigmoid()
    scores=torch.stack([image_score(F.interpolate(p[i:i+1,None],size=tuple(size),mode='bilinear',align_corners=False)[0,0]) for i,size in enumerate(sizes)])
    if not torch.isfinite(scores).all() or ((scores<1e-6)|(scores>1-1e-6)).any():raise FloatingPointError('Rank probability numerical guard would alter equation')
    return (torch.logit(scores)-b)/a,scores

def rank_evidence(model,out,sizes):
    # Reuse the exact already-computed cost/min graph, never rerun matching.
    distance=out['branch_distances'].mean(-1).reshape_as(out['logits'])
    e,s=evidence_from_distance(distance,sizes,model.rho,model.bias)
    if not torch.equal(s,out['scores']):raise AssertionError('Rank and ordinary forward scores differ')
    return e

def pair_rank(scores,labels):
    positive=labels.bool();negative=~positive
    if not positive.any() or not negative.any():return scores.sum()*0,0
    pos=scores[:,positive].min(0).values;neg=scores[:,negative].max(0).values
    return torch.relu(.1+neg[:,None]-pos[None,:]).mean(),len(pos)*len(neg)

def normalized_rank(e,labels,eta):
    if not math.isfinite(eta) or not 0<eta<=1:raise ValueError('Invalid frozen source eta')
    raw,n=pair_rank(e,labels)
    return raw/eta,raw,n

def paired_step(model,opt,scheduler,batch,phase,eta=1.):
    opt.zero_grad(set_to_none=True);lr=scheduler.prepare();refs=[];pixels=[];energies=[];scores=[]
    for i in range(2):
        state=model.prepare_context(batch['contexts'][i])
        refs.append(context_loss(state['normal_logits'],normality_targets(batch['context_masks'][i])))
        if phase=='joint':
            out=model.score_queries(state,batch['query'],batch['sizes'])
            pixels.append(pixel_terms(out['logits'],batch['query_masks'],batch['labels']))
            energies.append(rank_evidence(model,out,batch['sizes']));scores.append(out['scores'])
    ref=torch.stack(refs).mean();terms=dict(reference=ref)
    if phase=='joint':
        px=torch.stack([t['pixel'] for t in pixels]);pixel=.5*px.mean()+.5*px.max();e=torch.stack(energies)
        rank,raw,n=normalized_rank(e,batch['labels'],eta)
        terms.update(total=pixel+.2*ref+.1*rank,pixel=pixel,rank=rank,rank_raw=raw,
                     score_mean=torch.stack(scores).mean(),evidence_mean=e.mean())
        for i,t in enumerate(pixels):
            for k,v in t.items():terms[f'{k}_view{i}']=v
        pos=batch['labels'].bool();neg=~pos
        gap=float((e[:,pos].min(0).values[None,:]-e[:,neg].max(0).values[:,None]).mean().detach()) if n else None
    else:terms['total']=ref;n=0;gap=None
    if not all(torch.isfinite(v).all() for v in terms.values()):raise FloatingPointError('RR-Rank nonfinite loss')
    terms['total'].backward()
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    opt.step();scheduler.advance()
    return {**{k:float(v.detach()) for k,v in terms.items()},'lr':lr,'gradient_norm':float(grad),'rank_pairs':n,
            'eta':eta,'positive_queries':int(batch['labels'].sum()),'worst_pair_E_gap':gap,
            'a':float(F.softplus(model.rho).detach()),'b':float(model.bias.detach())}
