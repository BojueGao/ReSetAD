"""Label-blind mixed query queue; paired clean/contaminated contexts."""
import copy
from collections import Counter
import numpy as np
from src.data.episodes import CyclicQueue
from src.cache import canonical_hash


class PairSampler:
    def __init__(self,rows,neighbors,seed=17,cyclesize=503):
        self.rows=sorted(rows,key=lambda x:x['sample_id']);self.byid={r['sample_id']:r for r in self.rows}
        if len(self.byid)!=len(rows) or len({r['image_sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate source IDs/hashes')
        self.categories=sorted({r['category_id'] for r in rows});self.cyclesize=cyclesize
        self.rng=np.random.Generator(np.random.PCG64(seed));self.queues={};self.neighbors=neighbors
        for c in self.categories:
            pool=[r for r in self.rows if r['category_id']==c]
            for role,label in (('q',None),('normal',0),('anomaly',1)):
                subset=pool if label is None else [r for r in pool if r['label']==label]
                self.queues[f'{c}/{role}']=CyclicQueue(subset,int(self.rng.integers(0,2**63)))
            ids={r['sample_id'] for r in pool if r['label']==1}
            for sid in ids:
                if set(neighbors[sid])!=ids-{sid} or len(neighbors[sid])!=len(ids)-1:raise ValueError('Invalid source-neighbor inventory')
        self.contract=canonical_hash(dict(seed=seed,cyclesize=cyclesize,rows=[(r['sample_id'],r['image_sha256'],r['category_id'],r['label']) for r in self.rows],neighbors=neighbors))
        self.cycle=-1;self.phase=None;self.position=0;self.schedule=[];self.query_counts=Counter();self.context_counts=Counter()
    def begin_cycle(self,phase):
        if self.position!=len(self.schedule):raise ValueError('Cannot discard a partial cycle')
        if phase not in ('warmup','joint'):raise ValueError(phase)
        self.cycle+=1;self.phase=phase;self.position=0;self.query_counts=Counter();self.context_counts=Counter()
        for c in self.categories:self.queues[f'{c}/q'].reset()
        self.schedule=[str(c) for _ in range(self.cyclesize) for c in self.rng.permutation(self.categories)]
    def next(self):
        if self.position==len(self.schedule):raise StopIteration
        c=self.schedule[self.position];q=self.queues[f'{c}/q'].take_unique(4)
        ids={r['sample_id'] for r in q};hashes={r['image_sha256'] for r in q}
        clean=self.queues[f'{c}/normal'].take_unique(16,ids,hashes)
        k=int(self.rng.choice((0,2,4,8,12,16),p=(.1,.2,.3,.25,.1,.05)))
        coherent=bool(self.rng.integers(2)) if k else False
        if k==0:
            contaminated=self.queues[f'{c}/normal'].take_unique(16,ids|{r['sample_id'] for r in clean},hashes|{r['image_sha256'] for r in clean})
        else:
            retained=[clean[i] for i in self.rng.permutation(16)[:16-k]]
            if coherent:
                seed=self.queues[f'{c}/anomaly'].take_unique(1,ids,hashes)[0]
                candidates=[self.byid[s] for s in self.neighbors[seed['sample_id']] if s not in ids and self.byid[s]['image_sha256'] not in hashes]
                anomalous=[seed]+candidates[:k-1]
            else:anomalous=self.queues[f'{c}/anomaly'].take_unique(k,ids,hashes)
            contaminated=retained+anomalous
        self.rng.shuffle(clean);self.rng.shuffle(contaminated);self.rng.shuffle(q)
        for ctx in (clean,contaminated):
            if len(ctx)!=16 or len({r['image_sha256'] for r in ctx+q})!=20:raise AssertionError('C/Q duplicate or overlap')
        self.position+=1;self.query_counts.update(r['sample_id'] for r in q)
        self.context_counts.update(r['sample_id'] for r in clean+contaminated)
        return dict(category=c,query=q,contexts=[clean,contaminated],k=k,coherent=coherent,cycle=self.cycle,phase=self.phase)
    def audit_coverage(self):
        if self.position!=len(self.schedule):raise ValueError('Incomplete cycle')
        counts=[self.query_counts[r['sample_id']] for r in self.rows]
        if min(counts)<1:raise AssertionError('Source query coverage incomplete')
        return dict(cycle=self.cycle,phase=self.phase,steps=self.position,unique_query_images=len(counts),query_min=min(counts),query_max=max(counts),query_selections=sum(counts),query_counts=dict(self.query_counts),context_counts=dict(self.context_counts))
    def state_dict(self):
        return copy.deepcopy(dict(contract=self.contract,rng=self.rng.bit_generator.state,queues={k:v.state_dict() for k,v in self.queues.items()},cycle=self.cycle,phase=self.phase,position=self.position,schedule=self.schedule,query_counts=dict(self.query_counts),context_counts=dict(self.context_counts)))
    def load_state_dict(self,s):
        if s['contract']!=self.contract:raise ValueError('Sampler contract changed')
        for k in ('cycle','phase','position','schedule'):setattr(self,k,copy.deepcopy(s[k]))
        self.rng.bit_generator.state=copy.deepcopy(s['rng'])
        for k,v in s['queues'].items():self.queues[k].load_state_dict(v)
        self.query_counts=Counter(s['query_counts']);self.context_counts=Counter(s['context_counts'])
