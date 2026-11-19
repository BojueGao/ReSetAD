"""Independent persistent PCG64 queues; S001 authorized cycle-boundary reset."""
import copy
from dataclasses import dataclass
from collections import Counter
import numpy as np


class CyclicQueue:
    def __init__(self, rows, seed):
        self.rows=tuple(rows);self.rng=np.random.Generator(np.random.PCG64(seed))
        self.order=[];self.position=0;self.skips=0;self.reshuffles=0

    def reset(self):
        self.order=self.rng.permutation(len(self.rows)).tolist()
        self.position=0;self.reshuffles+=1

    def take_unique(self, n, excluded_ids=(), excluded_hashes=()):
        ids=set(excluded_ids);hashes=set(excluded_hashes);out=[]
        available={r['image_sha256'] for r in self.rows if r['sample_id'] not in ids and r['image_sha256'] not in hashes}
        if len(available)<n:raise ValueError('Insufficient unique eligible images')
        while len(out)<n:
            if self.position==len(self.order):self.reset()
            row=self.rows[self.order[self.position]];self.position+=1
            if row['sample_id'] in ids or row['image_sha256'] in hashes:
                self.skips+=1;continue
            out.append(row);ids.add(row['sample_id']);hashes.add(row['image_sha256'])
        return out

    def state_dict(self):
        return dict(order=list(self.order),position=self.position,skips=self.skips,
                    reshuffles=self.reshuffles,rng=copy.deepcopy(self.rng.bit_generator.state),
                    ids=[r['sample_id'] for r in self.rows])

    def load_state_dict(self,state):
        if state['ids']!=[r['sample_id'] for r in self.rows]:raise ValueError('Queue inventory changed')
        for key in ('order','position','skips','reshuffles'):setattr(self,key,copy.deepcopy(state[key]))
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])


@dataclass(frozen=True)
class Episode:
    category: str
    context: tuple
    query: tuple
    ratio: float
    cycle: int
    phase: str


class EpisodeSampler:
    def __init__(self, rows, seed=17, cyclesize=503):
        self.rows=rows;self.categories=sorted({r['category_id'] for r in rows})
        self.rng=np.random.Generator(np.random.PCG64(seed));self.queues={}
        for c in self.categories:
            for kind in ('q','c'):
                for label in (0,1):
                    key=f'{c}/{kind}/{label}'
                    self.queues[key]=CyclicQueue([r for r in rows if r['category_id']==c and r['label']==label],
                                                  int(self.rng.integers(0,2**63)))
        self.cyclesize=cyclesize;self.cycle=-1;self.phase=None;self.schedule=[];self.position=0
        self.query_counts=Counter();self.context_counts=Counter()

    def begin_cycle(self, phase):
        if self.position!=len(self.schedule):raise ValueError('Cannot reset inside a cycle')
        self.cycle+=1;self.phase=phase;self.position=0;self.query_counts=Counter()
        # S001: explicitly authorized, only at a new JOINT coverage cycle boundary.
        if phase=='joint':
            for c in self.categories:self.queues[f'{c}/q/0'].reset()
        self.schedule=[str(c) for _ in range(self.cyclesize) for c in self.rng.permutation(self.categories)]

    def next(self):
        if self.position>=len(self.schedule):raise StopIteration
        c=self.schedule[self.position]
        q=self.queues[f'{c}/q/0'].take_unique(2)+self.queues[f'{c}/q/1'].take_unique(2)
        nc=int(self.rng.choice([8,16,32],p=[.25,.5,.25]))
        ratio=float(self.rng.choice([0,.125,.25,.5,.75,1],p=[.1,.2,.3,.25,.1,.05]));na=round(nc*ratio)
        ids={r['sample_id'] for r in q};hashes={r['image_sha256'] for r in q}
        cn=self.queues[f'{c}/c/0'].take_unique(nc-na,ids,hashes)
        ca=self.queues[f'{c}/c/1'].take_unique(na,ids|{r['sample_id'] for r in cn},hashes|{r['image_sha256'] for r in cn})
        context=cn+ca;self.rng.shuffle(context);self.rng.shuffle(q)
        if {r['image_sha256'] for r in context}&{r['image_sha256'] for r in q}:raise AssertionError('C/Q leakage')
        if len({r['image_sha256'] for r in context+q})!=nc+4:raise AssertionError('Duplicate hash')
        self.position+=1
        if self.phase=='joint':self.query_counts.update(r['sample_id'] for r in q)
        self.context_counts.update(r['sample_id'] for r in context)
        return Episode(c,tuple(context),tuple(q),ratio,self.cycle,self.phase)

    def audit_coverage(self):
        if self.phase!='joint' or self.position!=len(self.schedule):raise ValueError('Need complete joint cycle')
        missing=[r['sample_id'] for r in self.rows if r['label']==0 and not self.query_counts[r['sample_id']]]
        if missing:raise AssertionError(f'Normal query coverage missing: {missing}')
        return dict(normal_min_count=min(self.query_counts[r['sample_id']] for r in self.rows if r['label']==0),
                    steps=self.position)

    def state_dict(self):
        return dict(rng=copy.deepcopy(self.rng.bit_generator.state),queues={k:q.state_dict() for k,q in self.queues.items()},
                    categories=self.categories,cyclesize=self.cyclesize,cycle=self.cycle,phase=self.phase,
                    schedule=list(self.schedule),position=self.position,query_counts=dict(self.query_counts),context_counts=dict(self.context_counts))

    def load_state_dict(self,s):
        if s['categories']!=self.categories or s['cyclesize']!=self.cyclesize:raise ValueError('Sampler protocol changed')
        self.rng.bit_generator.state=copy.deepcopy(s['rng'])
        for k,q in self.queues.items():q.load_state_dict(s['queues'][k])
        for k in ('cycle','phase','schedule','position'):setattr(self,k,copy.deepcopy(s[k]))
        self.query_counts=Counter(s['query_counts']);self.context_counts=Counter(s['context_counts'])
