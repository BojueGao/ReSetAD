"""Freeze source-only eta/task identities and controlled diagnostic IDs before GPU."""
from common import *
from sampling import PairSampler
from collections import Counter
import numpy as np

def diagnostic_tasks(rows,neighbors):
    rng=np.random.default_rng(1701);tasks=[];byid={r['sample_id']:r for r in rows}
    for category in sorted({r['category_id'] for r in rows}):
        pool=sorted([r for r in rows if r['category_id']==category],key=lambda r:r['sample_id'])
        for k in (0,2,4,8,12,16):
            for style in ('random','coherent'):
                q=[pool[i] for i in rng.choice(len(pool),4,replace=False)];qids={r['sample_id'] for r in q}
                normals=[r for r in pool if not r['label'] and r['sample_id'] not in qids]
                clean=[normals[i] for i in rng.choice(len(normals),16,replace=False)]
                if not k:
                    cleanids={r['sample_id'] for r in clean};eligible=[r for r in normals if r['sample_id'] not in cleanids]
                    contam=[eligible[i] for i in rng.choice(len(eligible),16,replace=False)]
                else:
                    anomaly=[r for r in pool if r['label'] and r['sample_id'] not in qids]
                    if style=='random':bad=[anomaly[i] for i in rng.choice(len(anomaly),k,replace=False)]
                    else:
                        seed=anomaly[int(rng.integers(len(anomaly)))];bad=[seed]+[byid[s] for s in neighbors[seed['sample_id']] if s not in qids][:k-1]
                    contam=[clean[i] for i in rng.permutation(16)[:16-k]]+bad
                rng.shuffle(clean);rng.shuffle(contam);rng.shuffle(q)
                for c in (clean,contam):
                    if len({r['image_sha256'] for r in c+q})!=20:raise AssertionError('Diagnostic duplicate')
                tasks.append(dict(id=f'{category}_k{k}_{style}',category=category,k=k,style=style,query=[r['sample_id'] for r in q],contexts=[[r['sample_id'] for r in c] for c in (clean,contam)]))
    return tasks

def main():
    parent_check();rows=source();neighbors=read(PARENT/'source_neighbors.json')
    if (RUN/'training_plan.json').exists():raise FileExistsError('Source plan already exists; verify and reuse')
    sampler=PairSampler(rows,neighbors);ids=[];coverage=[];allcounts=Counter();mixed=Counter()
    for cycle in range(5):
        sampler.begin_cycle('warmup' if cycle==0 else 'joint')
        for i in range(6036):
            e=sampler.next();ids.append(digest(episode_identity(e)))
            if cycle:
                allcounts[e['category']]+=1;mixed[e['category']]+=len({r['label'] for r in e['query']})==2
        coverage.append(sampler.audit_coverage());print(f'SOURCE PLAN cycle{cycle} complete',flush=True)
    eta={c:mixed[c]/allcounts[c] for c in sorted(allcounts)}
    if not all(0<x<=1 for x in eta.values()):raise ValueError('No eligible ranking tasks in category')
    write(RUN/'training_plan.json',dict(status='FROZEN',at=now(),seed=17,steps=30180,task_digests=ids,coverage=coverage,eta=eta,mixed_tasks=dict(mixed),joint_tasks=dict(allcounts),source_sha256=sha(ROOT/'outputs/manifests/source_full.jsonl'),neighbors_sha256=sha(PARENT/'source_neighbors.json')))
    tasks=diagnostic_tasks(rows,neighbors)
    write(RUN/'diagnostic_manifest.json',dict(status='FROZEN',seed=1701,tasks=tasks,count=len(tasks),variants=['learned','uniform'],scope='SOURCE mechanism audit; training-overlap, not held-out transfer; no model selection'))
    b=read(LEDGER)
    if any(a.get('seconds') is None for a in b['attempts']):raise ValueError('Open allocation: investigate before preparing')
    total=b['prior_gpu_hours']+sum(a['seconds']/3600 for a in b['attempts'])
    if abs(total-82.07682615532477)>1e-9:raise ValueError('Unexpected prior GPU total; reconcile explicitly')
    write(RUN/'budget_prefix_snapshot.json',dict(ledger=b,sha256=sha(LEDGER),total=total,at=now()))
    print('SOURCE PLAN AND DIAGNOSTICS FROZEN '+str(eta),flush=True)
if __name__=='__main__':main()
