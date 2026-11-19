import math
import torch
from torch import nn
from src.models.relations import branches,fixed_anchors,imagewise_min,quantile_mix,jackknife_quantile_mix,robust_stats


class FixedBaseline(nn.Module):
    def __init__(self,kind,node_chunk=64):
        super().__init__()
        if kind not in ('B0','B1'):raise ValueError(kind)
        self.kind=kind;self.node_chunk=node_chunk

    def prepare_context(self,context_features,valid_mask=None):
        nc,_,h,w,_=context_features.shape
        anchors=fixed_anchors(branches(context_features),h,w)
        state={'anchors':anchors,'valid':valid_mask}
        if self.kind=='B1':
            dc,_=imagewise_min(anchors,anchors,exclude_same_image=True,node_chunk=self.node_chunk,valid_mask=valid_mask)
            alpha=anchors.new_zeros(12,5);alpha[:,1]=1
            xc=quantile_mix(dc,alpha)
            mu,sigma,_=robust_stats(xc.reshape(-1,12),torch.ones_like(xc).reshape(-1,12),None if valid_mask is None else valid_mask.flatten())
            state.update(alpha=alpha,mu=mu,sigma=sigma)
        return state

    def score_queries(self,state,query_features):
        n,_,h,w,_=query_features.shape
        distances,_=imagewise_min(branches(query_features),state['anchors'],node_chunk=self.node_chunk,valid_mask=state['valid'])
        if self.kind=='B0':scores=distances.min(-1).values.mean(-1)
        else:scores=((jackknife_quantile_mix(distances,state['alpha'])-state['mu'])/state['sigma']).mean(-1)
        raw_map=scores.reshape(n,h,w)
        return {'raw_map':raw_map,'raw_image_score':scores.topk(math.ceil(.01*h*w),dim=-1).values.mean(-1),
                'threshold_status':'NA_NO_CALIBRATED_THRESHOLD'}

    def forward(self,context_features,query_features,valid_mask=None):
        return self.score_queries(self.prepare_context(context_features,valid_mask),query_features)
