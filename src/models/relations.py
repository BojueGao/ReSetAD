"""Exact FP32 relations, with gradients through the penalty before argmin."""
import math
import torch
import torch.nn.functional as F

LEVELS=(0.,.10,.25,.50,.90)


def branches(features):
    """[N,4,H,W,D] -> [N,H*W,12,D], layer-major, scale-minor."""
    if features.ndim!=5 or features.shape[1]!=4:raise ValueError('Expected N,4,H,W,D frozen features')
    if features.requires_grad:raise ValueError('Frozen input features must not require gradients')
    n,l,h,w,d=features.shape
    x=features.float().permute(0,1,4,2,3).reshape(n*l,d,h,w)
    values=[]
    for r in (1,3,5):
        y=F.avg_pool2d(F.pad(x,(r//2,)*4,mode='reflect'),r,stride=1) if r>1 else x
        values.append(F.normalize(y,dim=1,eps=1e-6).reshape(n,l,d,h,w))
    return torch.stack(values,dim=2).permute(0,4,5,1,2,3).reshape(n,h*w,12,d)


def fixed_anchors(full, height, width):
    if height<4 or width<4:raise ValueError('Reference grid too small')
    n,p,b,d=full.shape
    if p!=height*width:raise ValueError('Grid mismatch')
    return full.reshape(n,height,width,b,d)[:,1::2,1::2].reshape(n,-1,b,d)


def imagewise_min(nodes, anchors, weights=None, lam=None, *, exclude_same_image=False,
                  node_chunk=64, valid_mask=None):
    """nodes [N,P,B,D], anchors [C,A,B,D]; final axis is reference IMAGE.

    Raw frozen distances are computed in bounded chunks. Only the selected-index
    min graph is retained, preserving autograd through weights and lambda.
    """
    n,p,b,d=nodes.shape;nc,a,bb,dd=anchors.shape
    if (b,d)!=(bb,dd) or nc<2:raise ValueError('Invalid branch/reference tensor')
    if exclude_same_image and n!=nc:raise ValueError('Self-image exclusion requires aligned images')
    if node_chunk<1:raise ValueError('node_chunk must be positive')
    valid=torch.ones(nc,a,dtype=torch.bool,device=anchors.device) if valid_mask is None else valid_mask
    if valid.shape!=(nc,a) or valid.dtype!=torch.bool or not valid.any(-1).all():
        raise ValueError('Each reference image needs valid anchors')
    refs=anchors.detach().permute(2,0,1,3).reshape(b,nc*a,d).transpose(1,2).contiguous()
    queries=nodes.detach().permute(2,0,1,3).reshape(b,n*p,d)
    if weights is None:
        penalty=anchors.new_zeros(b,nc,a)
    else:
        if weights.shape!=(nc,a,b) or lam.shape!=(b,):raise ValueError('Penalty broadcast mismatch')
        penalty=lam[:,None,None]*(1-weights.permute(2,0,1))
    values=[];matches=[]
    for start in range(0,n*p,node_chunk):
        with torch.no_grad():
            raw=1-torch.bmm(queries[:,start:start+node_chunk],refs).clamp(-1,1)
            raw=raw.reshape(b,-1,nc,a)
        costs=(raw+penalty[:,None]).masked_fill(~valid[None,None],float('inf'))
        minimum,index=costs.min(-1)
        minimum=minimum.permute(1,0,2);index=index.permute(1,0,2)
        if exclude_same_image:
            owner=torch.arange(start,min(start+node_chunk,n*p),device=nodes.device)//p
            other=torch.arange(nc,device=nodes.device).expand(len(owner),nc)
            other=other[other!=owner[:,None]].reshape(len(owner),nc-1)
            selection=other[:,None].expand(-1,b,-1)
            minimum=minimum.gather(-1,selection);index=index.gather(-1,selection)
        values.append(minimum);matches.append(index)
    return torch.cat(values).reshape(n,p,b,-1),torch.cat(matches).reshape(n,p,b,-1)


def quantiles(distances):
    levels=torch.tensor(LEVELS,dtype=distances.dtype,device=distances.device)
    return torch.quantile(distances,levels,dim=-1,interpolation='linear').movedim(0,-1)


def quantile_mix(distances,alpha):
    return (quantiles(distances)*alpha).sum(-1)


def relation_input(distances, nc):
    q=quantiles(distances).flatten(-2)
    return torch.cat((q,torch.full_like(q[...,:1],math.log(nc)/math.log(32))),-1)


def jackknife_quantile_mix(distances,alpha):
    """Exact mean of all leave-one-reference quantiles, sorting just once.

    For the h-th order statistic of a deletion, h+1 deletions choose sorted[h+1]
    and N-h-1 choose sorted[h]. Linear quantile interpolation commutes with mean.
    Validated against an independent explicit deletion/quantile loop, incl grads.
    """
    nc=distances.shape[-1]
    if nc<2:raise ValueError('Jackknife needs >=2 reference images')
    ordered=distances.sort(dim=-1,stable=True).values
    positions=torch.tensor(LEVELS,device=distances.device,dtype=distances.dtype)*(nc-2)
    lo=positions.floor().long();hi=positions.ceil().long();fraction=positions-lo
    def mean_order(indices):
        return ((nc-indices-1)*ordered[...,indices]+(indices+1)*ordered[...,indices+1])/nc
    mixed=mean_order(lo)*(1-fraction)+mean_order(hi)*fraction
    return (mixed*alpha).sum(-1)


def robust_stats(x,w,valid=None):
    if x.shape!=w.shape or x.ndim!=2:raise ValueError('Expected matching [nodes,branches]')
    if valid is not None:x=x[valid];w=w[valid]
    if len(x)==0 or not (w>0).all():raise ValueError('No positive calibration weights')
    median=torch.quantile(x,.5,dim=0,interpolation='linear')
    mad=torch.quantile((x-median).abs(),.5,dim=0,interpolation='linear')+.01
    v=median+3*mad*torch.tanh((x-median)/(3*mad))
    mu=(w*v).sum(0)/w.sum(0)
    sigma=torch.sqrt((w*(v-mu).square()).sum(0)/w.sum(0)+.01**2)
    return mu,sigma,dict(median=median,mad=mad,clipped=v)
