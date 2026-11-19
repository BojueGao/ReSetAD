"""No annotation argument or lookup: grouping consumes ImageRecord only."""
import hashlib
import json
from collections import defaultdict
import numpy as np
from .manifest import ImageRecord


def groups(records, dataset_id, pool_id, seed=101):
    if not all(type(r) is ImageRecord for r in records):raise TypeError('ImageRecord stream required')
    rows=sorted(records,key=lambda r:r.sample_id)
    if len({r.sample_id for r in rows})!=len(rows):raise ValueError('Duplicate query IDs')
    if len(rows)<3:raise ValueError('NO_VALID_CONTEXT')
    material=json.dumps([dataset_id,pool_id,int(seed)],separators=(',',':')).encode()
    stream_seed=int.from_bytes(hashlib.sha256(material).digest(),'big')
    rng=np.random.Generator(np.random.PCG64(stream_seed))
    rows=[rows[int(i)] for i in rng.permutation(len(rows))]
    result=[rows[i:i+20] for i in range(0,len(rows),20)]
    if len(result)>1 and len(result[-1])<12:result[-2].extend(result.pop())
    return result


def rounds(group):
    chunk=min(4,len(group)-2) if len(group)<12 else 4
    for start in range(0,len(group),chunk):
        q=group[start:start+chunk];qids={r.sample_id for r in q}
        candidates=[r for r in group if r.sample_id not in qids]
        buckets=defaultdict(list);context_by_key={}
        for query in q:
            used_hashes=set();eligible=[]
            for r in sorted(candidates,key=lambda r:r.sample_id):
                if r.sha256==query.sha256 or r.sha256 in used_hashes:continue
                if query.optional_case_id is not None and r.optional_case_id==query.optional_case_id:continue
                eligible.append(r);used_hashes.add(r.sha256)
            key=tuple(r.sample_id for r in eligible)
            buckets[key].append(query);context_by_key[key]=eligible
        for key,queries in buckets.items():
            yield dict(query=queries,context=context_by_key[key],
                       status='READY' if len(key)>=2 else 'NO_VALID_CONTEXT')
