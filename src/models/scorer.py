import math
import torch
from torch import nn
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,96))

    def forward(self,h):
        v=self.net(h)
        return v[:60].reshape(12,5).softmax(-1),.25*v[60:72].sigmoid(),v[72:84].softmax(-1),4*v[84:96].sigmoid()


class PositiveSmoother(nn.Module):
    def __init__(self):
        super().__init__();self.kernel_logits=nn.Parameter(torch.zeros(12,9));self.gates=nn.Parameter(torch.empty(12))

    def forward(self,t):
        weights=self.kernel_logits.softmax(-1).reshape(12,1,3,3)
        gamma=.25*self.gates.sigmoid()[None,:,None,None]
        smooth=F.conv2d(F.pad(t,(1,1,1,1),mode='reflect'),weights,groups=12)
        return (1-gamma)*t+gamma*smooth


class MonotoneHead(nn.Module):
    def __init__(self):
        super().__init__();self.A=nn.Parameter(torch.empty(12,8));self.V=nn.Parameter(torch.empty(12,8))
        self.C=nn.Parameter(torch.empty(12,8));self.beta0=nn.Parameter(torch.tensor(-1.))

    def phi(self,t):
        ap=F.softplus(self.A)[None,:,None,None,:];vp=F.softplus(self.V)[None,:,None,None,:]
        cp=self.C[None,:,None,None,:]
        return (ap*(F.softplus(vp*t[...,None]+cp)-F.softplus(cp))).sum(-1)

    def forward(self,t,omega,tau):
        contribution=omega[None,:,None,None]*self.phi(t-tau[None,:,None,None])
        logits=self.beta0+contribution.sum(1)
        k=math.ceil(.01*logits.shape[-1]*logits.shape[-2])
        image_logit=logits.flatten(1).topk(k,dim=1).values.mean(1)
        return logits,image_logit,contribution


def pixel_maps(logits,sizes):
    if len(logits)!=len(sizes):raise ValueError('One original size per query')
    return [F.interpolate(l.sigmoid()[None,None],size=tuple(hw),mode='bilinear',align_corners=False)[0,0]
            for l,hw in zip(logits,sizes)]
