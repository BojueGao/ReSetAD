import torch
import torch.nn.functional as F


def normality_targets(masks):
    if masks.ndim!=3 or any(v%14 for v in masks.shape[-2:]):raise ValueError('Source mask sides must be multiples of14')
    patch=F.max_pool2d(masks[:,None].float(),14,stride=14)
    labels=[]
    for r in (1,3,5):
        support=F.max_pool2d(F.pad(patch,(r//2,)*4,mode='reflect'),r,stride=1) if r>1 else patch
        labels.append((1-support[:,0,1::2,1::2]).flatten(1))
    return torch.stack(labels,-1).repeat(1,1,4)


def context_loss(logits,targets,valid=None):
    if logits.shape!=targets.shape:raise ValueError('Context target shape mismatch')
    if valid is None:valid=torch.ones(logits.shape[:-1],dtype=torch.bool,device=logits.device)
    bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    losses=[]
    for b in range(logits.shape[-1]):
        states=[bce[...,b][valid&(targets[...,b]==s)] for s in (0,1)]
        states=[v.mean() for v in states if v.numel()]
        if not states:raise ValueError('No valid context supervision')
        losses.append(torch.stack(states).mean())
    return torch.stack(losses).mean()


def query_losses(logits,image_logit,masks,labels):
    probs=F.interpolate(logits.sigmoid()[:,None],size=tuple(masks.shape[-2:]),mode='bilinear',align_corners=False)[:,0]
    p=probs.clamp(1e-6,1-1e-6)
    # Clamp only logarithm arguments (§26), not the focal weighting factors.
    focal=(-.75*masks*(1-probs).square()*p.log()-.25*(1-masks)*probs.square()*torch.log1p(-p)).mean()
    positive=labels==1
    if positive.any():
        if not (masks[positive].flatten(1).sum(1)>0).all():raise ValueError('Empty anomaly query mask')
        pred=probs[positive].flatten(1);gt=masks[positive].flatten(1)
        dice=(1-(2*(pred*gt).sum(1)+1)/(pred.sum(1)+gt.sum(1)+1)).mean()
    else:dice=probs.sum()*0
    image=F.binary_cross_entropy_with_logits(image_logit,labels.float())
    pixel=focal+.5*dice
    return {'focal':focal,'dice':dice,'pixel':pixel,'image':image,'query':pixel+.5*image}


def criterion(output,context_masks,query_masks,query_labels):
    result=query_losses(output.logits,output.image_logit,query_masks,query_labels)
    result['context']=context_loss(output.normal_logits,normality_targets(context_masks),output.context.valid)
    result['total']=result['query']+.2*result['context']
    return result
