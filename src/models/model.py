import math
from dataclasses import dataclass
import torch
from torch import nn
from .relations import branches,fixed_anchors,imagewise_min,relation_input,quantile_mix,jackknife_quantile_mix,robust_stats
from .set_encoder import SetEncoder
from .scorer import Controller,PositiveSmoother,MonotoneHead


@dataclass
class ContextState:
    anchors: torch.Tensor
    normal_logits: torch.Tensor
    w: torch.Tensor
    alpha: torch.Tensor
    lam: torch.Tensor
    omega: torch.Tensor
    tau: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    valid: torch.Tensor
    h: torch.Tensor
    debug: dict


@dataclass
class Outputs:
    logits: torch.Tensor
    image_logit: torch.Tensor
    normal_logits: torch.Tensor
    contributions: torch.Tensor
    matches: torch.Tensor
    context: ContextState


class VSIAD(nn.Module):
    def __init__(self,node_chunk=64,backbone=None):
        super().__init__();self.backbone=backbone;self.node_chunk=node_chunk
        self.encoder=SetEncoder();self.normality=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,12))
        self.controller=Controller();self.smoother=PositiveSmoother();self.head=MonotoneHead()
        self.initialize()

    def initialize(self):
        # Do not touch any registered frozen backbone parameters.
        for root in (self.encoder,self.normality,self.controller,self.smoother,self.head):
            for module in root.modules():
                if isinstance(module,nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:nn.init.zeros_(module.bias)
                if isinstance(module,nn.MultiheadAttention):
                    for weight in module.in_proj_weight.chunk(3,dim=0):nn.init.xavier_uniform_(weight)
                    nn.init.zeros_(module.in_proj_bias)
                if isinstance(module,nn.LayerNorm):nn.init.ones_(module.weight);nn.init.zeros_(module.bias)
        nn.init.normal_(self.encoder.isab1.tokens,0,.02);nn.init.normal_(self.encoder.isab2.tokens,0,.02)
        nn.init.normal_(self.encoder.seed,0,.02)
        with torch.no_grad():
            self.controller.net[-1].weight.zero_();self.controller.net[-1].bias.zero_()
            self.controller.net[-1].bias[60:72]=math.log(.08/.92)
            self.controller.net[-1].bias[84:96]=math.log(.25/.75)
            self.smoother.kernel_logits.zero_();self.smoother.kernel_logits[:,4]=2
            self.smoother.gates.fill_(math.log(.2/.8))
            self.head.A.fill_(math.log(math.expm1(.25)));self.head.V.fill_(math.log(math.expm1(1)))
            self.head.C.copy_(torch.linspace(-1,1,8).expand(12,8));self.head.beta0.fill_(-1)

    def train(self,mode=True):
        super().train(mode)
        if self.backbone is not None:self.backbone.eval()
        return self

    def set_stage(self,stage):
        if stage not in ('warmup','joint'):raise ValueError(stage)
        for name,p in self.named_parameters():
            p.requires_grad_(not name.startswith('backbone.') and (stage=='joint' or name.startswith(('encoder.embed.','encoder.isab1.','encoder.isab2.','normality.'))))

    def _nodes(self,features,valid_mask):
        full=branches(features);n,_,_,_=full.shape;h,w=features.shape[2:4]
        anchors=fixed_anchors(full,h,w)
        valid=torch.ones(anchors.shape[:2],dtype=torch.bool,device=features.device) if valid_mask is None else valid_mask
        with torch.no_grad():
            d0,_=imagewise_min(anchors,anchors,exclude_same_image=True,node_chunk=self.node_chunk,valid_mask=valid)
            u=relation_input(d0,n).reshape(-1,61)
        z=self.encoder.nodes(u,valid)
        normal=self.normality(z[0]).reshape(n,-1,12)
        return anchors,valid,z,normal

    def context_normality(self,context_features,valid_mask=None):
        return self._nodes(context_features,valid_mask)[-1]

    def prepare_context(self,context_features,valid_mask=None):
        anchors,valid,z,normal=self._nodes(context_features,valid_mask)
        w=.01+.98*normal.sigmoid();h=self.encoder.pool(z,valid)
        alpha,lam,omega,tau=self.controller(h)
        dc,_=imagewise_min(anchors,anchors,w,lam,exclude_same_image=True,node_chunk=self.node_chunk,valid_mask=valid)
        xc=quantile_mix(dc,alpha)
        mu,sigma,debug=robust_stats(xc.reshape(-1,12),w.reshape(-1,12),valid.flatten())
        return ContextState(anchors,normal,w,alpha,lam,omega,tau,mu,sigma,valid,h,debug)

    def score_queries(self,state,query_features):
        fq=branches(query_features);n= len(fq);h,w=query_features.shape[2:4]
        dq,matches=imagewise_min(fq,state.anchors,state.w,state.lam,node_chunk=self.node_chunk,valid_mask=state.valid)
        xq=jackknife_quantile_mix(dq,state.alpha)
        t=((xq-state.mu)/state.sigma).permute(0,2,1).reshape(n,12,h,w)
        logits,image_logit,contributions=self.head(self.smoother(t),state.omega,state.tau)
        return Outputs(logits,image_logit,state.normal_logits,contributions,matches,state)

    def forward(self,context_features,query_features,valid_mask=None):
        return self.score_queries(self.prepare_context(context_features,valid_mask),query_features)
