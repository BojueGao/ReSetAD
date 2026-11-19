"""RR v1: context-only reference reliability, fixed distance rule, shared readout."""
import math
from src.resolution import INPUT_SIZE
import torch
from torch import nn
import torch.nn.functional as F
from src.models.relations import branches, fixed_anchors, imagewise_min


@torch.no_grad()
def edge_features(anchors, height, width):
    """[C,A,12,D] -> [C*A,C-1,84]. No labels or raw embeddings enter the net."""
    c,a,b,d=anchors.shape
    if c<2 or a!=height*width or a<4 or b!=12:raise ValueError('Invalid reference grid')
    refs=anchors.permute(2,0,1,3).reshape(b,c*a,d).transpose(1,2).contiguous()
    distances=[];indices=[];means=[];gaps=[]
    for i in range(c):
        raw=(1-torch.bmm(anchors[i].permute(1,0,2),refs).clamp(-1,1)).reshape(b,a,c,a)
        nearest,idx=raw.min(-1)  # stable first anchor for an exact tie
        four=raw.topk(4,dim=-1,largest=False,sorted=True).values
        distances.append(nearest.permute(1,0,2));indices.append(idx.permute(1,0,2))
        means.append(four.mean(-1).permute(1,0,2));gaps.append((four[...,1]-four[...,0]).permute(1,0,2))
    dist=torch.stack(distances);idx=torch.stack(indices);mean=torch.stack(means);gap=torch.stack(gaps)
    own=torch.arange(c,device=anchors.device)[:,None,None,None]
    ref=torch.arange(c,device=anchors.device)[None,None,None,:]
    branch=torch.arange(b,device=anchors.device)[None,None,:,None]
    mutual=(idx[ref,idx,branch,own]==torch.arange(a,device=anchors.device)[None,:,None,None]).float()
    grid=dist.reshape(c,height,width,b,c);variation=torch.zeros_like(grid);counts=torch.zeros((height,width),device=grid.device)
    if height>1:
        delta=(grid[:,1:]-grid[:,:-1]).abs()
        variation[:,1:]+=delta;variation[:,:-1]+=delta;counts[1:]+=1;counts[:-1]+=1
    if width>1:
        delta=(grid[:,:,1:]-grid[:,:,:-1]).abs()
        variation[:,:,1:]+=delta;variation[:,:,:-1]+=delta;counts[:,1:]+=1;counts[:,:-1]+=1
    variation=(variation/counts.clamp_min(1)[None,:,:,None,None]).reshape_as(dist)
    others=torch.arange(c,device=anchors.device).expand(c,c)
    others=others[others!=torch.arange(c,device=anchors.device)[:,None]].reshape(c,c-1)
    sel=others[:,None,None,:].expand(c,a,b,c-1)
    take=lambda x:x.gather(-1,sel)
    nearest=take(dist)
    # Average ranks, smallest=0; for the one-reference boundary rank=0.
    ranks=((nearest.unsqueeze(-2)<nearest.unsqueeze(-1)).sum(-1)
           +.5*((nearest.unsqueeze(-2)==nearest.unsqueeze(-1)).sum(-1)-1))/max(1,c-2)
    features=torch.stack((nearest/2,take(mean)/2,take(gap)/2,ranks,take(mutual),take(variation)/2,
                          torch.full_like(nearest,math.log(c)/math.log(32))),dim=-1)
    return features.permute(0,1,3,2,4).reshape(c*a,c-1,b*7)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.attn=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.ln1=nn.LayerNorm(64);self.ln2=nn.LayerNorm(64)
        self.ff=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
    def forward(self,x):
        x=self.ln1(x+self.attn(x,x,x,need_weights=False)[0]);return self.ln2(x+self.ff(x))


class Reliability(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Sequential(nn.Linear(84,64),nn.GELU(),nn.LayerNorm(64))
        self.blocks=nn.Sequential(AttentionBlock(),AttentionBlock())
        self.seed=nn.Parameter(torch.zeros(1,1,64));self.pool=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.norm=nn.LayerNorm(64);self.head=nn.Linear(64,12)
    def forward(self,edges):
        x=self.blocks(self.embed(edges));q=self.seed.expand(len(x),-1,-1)
        return self.head(self.norm(q+self.pool(q,x,x,need_weights=False)[0])[:,0])


def image_score(probability):
    flat=probability.flatten(-2);return flat.topk(max(1,math.ceil(flat.shape[-1]*.01)),dim=-1).values.mean(-1)


class RR(nn.Module):
    def __init__(self,node_chunk=64):
        super().__init__();self.reliability=Reliability();self.rho=nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        self.bias=nn.Parameter(torch.tensor(0.));self.node_chunk=node_chunk
    def set_stage(self,phase):
        if phase not in ('warmup','joint'):raise ValueError(phase)
        self.rho.requires_grad_(phase=='joint');self.bias.requires_grad_(phase=='joint')
    def prepare_context(self,features):
        h,w=features.shape[2:4]
        anchors=fixed_anchors(branches(features),h,w)
        edges=edge_features(anchors,h//2,w//2)
        logits=self.reliability(edges).reshape(len(features),anchors.shape[1],12)
        return dict(anchors=anchors,normal_logits=logits,weights=logits.sigmoid(),height=h,width=w)
    def score_queries(self,state,query,sizes=None):
        h,w=query.shape[2:4];q=branches(query)
        values,matches=imagewise_min(q,state['anchors'],state['weights'],q.new_full((12,),.25),node_chunk=self.node_chunk)
        best=values.topk(2,dim=-1,largest=False,sorted=True)
        per_branch=best.values.mean(-1);distance=per_branch.mean(-1).reshape(len(query),h,w)
        logits=F.softplus(self.rho)*distance+self.bias
        probs=logits.sigmoid()
        if sizes is None:sizes=[(INPUT_SIZE,INPUT_SIZE)]*len(query)
        maps=[F.interpolate(probs[i:i+1,None],size=tuple(size),mode='bilinear',align_corners=False)[0,0] for i,size in enumerate(sizes)]
        scores=torch.stack([image_score(p) for p in maps])
        return dict(logits=logits,scores=scores,maps=maps,branch_distances=per_branch,
                    reference_images=best.indices,matches=matches)


def fuse_maps(round_maps):
    if len(round_maps)!=3 or len({tuple(x.shape) for x in round_maps})!=1:raise ValueError('Three aligned native maps required')
    mean=torch.stack(round_maps).mean(0)
    return mean,image_score(mean)
