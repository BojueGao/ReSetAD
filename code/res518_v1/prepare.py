"""Byte audit of existing source data; no model execution or new sampling."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import *

def main():
    parent_check();rows=source()
    for i,r in enumerate(rows):
        assert sha(r['path'])==r['image_sha256'],r['sample_id']
        if r['label']:assert sha(r['mask_path'])==r['mask_sha256']
        if (i+1)%1000==0:print('SOURCE_BYTES',i+1,flush=True)
    write(RUN/'source_preflight.json',dict(status='READY',images=len(rows),source_bytes_verified=True,
         tasks_sha256=sha(RUN/'training_plan.json'),diagnostics_sha256=sha(RUN/'diagnostic_manifest.json'),time=now()))
if __name__=='__main__':main()
