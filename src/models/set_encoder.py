import torch
from torch import nn


class MAB(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn=nn.MultiheadAttention(128,4,dropout=0,batch_first=True)
        self.ln1=nn.LayerNorm(128);self.ln2=nn.LayerNorm(128)
        self.ff=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))

    def forward(self,x,y,valid=None):
        mask=None if valid is None else ~valid
        h=self.ln1(x+self.attn(x,y,y,key_padding_mask=mask,need_weights=False)[0])
        return self.ln2(h+self.ff(h))


class ISAB(nn.Module):
    def __init__(self):
        super().__init__();self.tokens=nn.Parameter(torch.empty(1,16,128))
        self.inner=MAB();self.outer=MAB()

    def forward(self,x,valid=None):
        return self.outer(x,self.inner(self.tokens,x,valid))


class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(61,128),nn.GELU(),nn.LayerNorm(128))
        self.isab1=ISAB();self.isab2=ISAB();self.pma=MAB()
        self.seed=nn.Parameter(torch.empty(1,1,128))

    def nodes(self,u,valid=None):
        x=self.embed(u)[None]
        mask=None if valid is None else valid.reshape(1,-1)
        return self.isab2(self.isab1(x,mask),mask)

    def pool(self,z,valid=None):
        return self.pma(self.seed,z,None if valid is None else valid.reshape(1,-1))[0,0]
