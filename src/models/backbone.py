from pathlib import Path
import sys
import torch
from torch import nn


PRESETS={'dinov2_vitb14':(768,(2,5,8,11)), 'dinov2_vitl14':(1024,(5,11,17,23))}

class FrozenBackbone(nn.Module):
    def __init__(self,repo,weights,backbone_id='dinov2_vitl14',layer_indices=None):
        super().__init__()
        if backbone_id not in PRESETS:raise ValueError('Only non-register DINOv2 B/L are supported')
        self.dimension,self.layers=PRESETS[backbone_id]
        if layer_indices is not None and tuple(layer_indices)!=self.layers:raise ValueError('Frozen layer selection changed')
        sys.path.insert(0,str(Path(repo).resolve()))
        from dinov2.hub import backbones
        self.encoder=getattr(backbones,backbone_id)(pretrained=False)
        state=torch.load(weights,map_location='cpu',weights_only=True)
        if any('register' in key for key in state):raise ValueError('Register variant is forbidden')
        self.encoder.load_state_dict(state,strict=True)
        for p in self.parameters():p.requires_grad_(False)
        self.eval()

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self,images):
        if images.ndim!=4 or images.shape[1]!=3 or any(v%14 for v in images.shape[2:]):raise ValueError('RGB image sides must be multiples of14')
        height,width=images.shape[2]//14,images.shape[3]//14
        result=self.encoder.get_intermediate_layers(images,n=list(self.layers),reshape=False,return_class_token=False,norm=True)
        if any(tuple(t.shape)!=(len(images),height*width,self.dimension) for t in result):raise ValueError('Unexpected DINO feature shape')
        return torch.stack(result,dim=1).reshape(len(images),4,height,width,self.dimension)
