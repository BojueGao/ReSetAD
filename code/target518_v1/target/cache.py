"""S002 cache uses a distinct transform hash and per-key interprocess write locks."""
from pathlib import Path
import fcntl
import torch
from src.cache import FeatureCache,cache_header,canonical_hash
from src.evidence import sha256
from target.data import DECODER,image_tensor

def header(config):
    value=cache_header(config)
    value['input_transform_hash']=canonical_hash({'preprocess':config['preprocess'],'decoder':DECODER})
    value['encoder_microbatch']=1
    return value

class TargetCache(FeatureCache):
    def ensure(self,rows):
        # Single shared extraction job is normal; locks protect explicit recovery or concurrent readers.
        for row in rows:
            path=self.path(row)
            with path.with_suffix('.lock').open('a+') as guard:
                fcntl.flock(guard,fcntl.LOCK_EX)
                if path.exists():continue
                if self.backbone is None:raise FileNotFoundError('Target feature cache must be complete before inference')
                assert sha256(row['path'])==row['image_sha256']
                x=image_tensor(row['path'])[None].to(self.device)
                with torch.no_grad():features=self.backbone(x).to(getattr(torch,self.header['dtype'])).cpu()
                temporary=path.with_suffix('.tmp')
                torch.save({'metadata':{**self.header,'sample_sha':row['image_sha256']},'features':features[0].clone()},temporary)
                temporary.replace(path)
