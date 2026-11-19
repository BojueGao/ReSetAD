"""CPU-only terminal audit, executed only after training GPU process exits."""
from common import ROOT,RUN,read,write,sha,source,contract,now,digest
import torch,json
from collections import Counter
from model import RR
from src.engine import state_hash

def main():
    frozen=contract();done=read(RUN/'training_complete.json');latest=read(RUN/'latest_checkpoint.json')
    if done['contract']!=frozen or done['step']!=30180 or latest['step']!=30180:raise AssertionError('Incomplete training')
    if sha(latest['path'])!=latest['sha256'] or sha(RUN/'inference_model.pt')!=done['export_sha256']:raise AssertionError('Checkpoint/export changed')
    checkpoint=torch.load(latest['path'],map_location='cpu',weights_only=False)
    export=torch.load(RUN/'inference_model.pt',map_location='cpu',weights_only=False)
    if checkpoint['scheduler']!={'horizon':24144,'steps':24144}:raise AssertionError('Wrong scheduler horizon/steps')
    if checkpoint['metadata']['contract']!=frozen or export['contract']!=frozen:raise AssertionError('Contract mismatch')
    m=RR();m.load_state_dict(checkpoint['model']);last=state_hash(m);m.load_state_dict(export['model'])
    if last!=state_hash(m) or last!=export['state_hash']:raise AssertionError('Export is not final state')
    entries={};trace_hashes={}
    for p in sorted(RUN.glob('episodes_from_*.jsonl')):
        trace_hashes[p.name]=sha(p)
        for line in p.read_text().splitlines():
            e=json.loads(line)
            if e['step'] in entries:raise AssertionError('Replay requires explicit attempt-aware audit; not silently counted')
            entries[e['step']]=e
    if sorted(entries)!=list(range(1,30181)):raise AssertionError('Missing actual optimizer update logs')
    plan=read(RUN/'training_plan.json');rows={r['sample_id']:r for r in source()};joint=Counter();mixed=Counter()
    for step,e in sorted(entries.items()):
        identity=dict(category=e['category'],cycle=e['cycle'],phase=e['phase'],k=e['k'],coherent=e['coherent'],query=e['query_ids'],contexts=e['context_ids'])
        if digest(identity)!=e['task_digest'] or e['task_digest']!=plan['task_digests'][step-1]:raise AssertionError('Actual task differs from frozen plan')
        if e['query_labels']!=[rows[s]['label'] for s in e['query_ids']]:raise AssertionError('Source query label mismatch')
        if e['eta']!=plan['eta'][e['category']]:raise AssertionError('Wrong ranking normalization')
        for c in e['context_ids']:
            if len({rows[s]['image_sha256'] for s in c+e['query_ids']})!=20:raise AssertionError('Within-view duplicate image hash')
        if e['phase']=='joint':
            joint[e['category']]+=1;mixed[e['category']]+=len(set(e['query_labels']))==2
            expected=e['pixel']+.2*e['reference']+.1*e['rank']
            if abs(e['total']-expected)>2e-6 or abs(e['rank']-e['rank_raw']/e['eta'])>2e-6:raise AssertionError('Loss equation mismatch')
    if dict(joint)!=plan['joint_tasks'] or dict(mixed)!=plan['mixed_tasks']:raise AssertionError('Actual eta task counts mismatch')
    ids={r['sample_id'] for r in source()};coverage=[]
    for cycle in range(5):
        es=[e for e in entries.values() if e['cycle']==cycle];counts=Counter(i for e in es for i in e['query_ids'])
        if len(es)!=6036 or set(counts)!=ids:raise AssertionError('Cycle incomplete')
        saved=read(RUN/'coverage'/f'cycle_{cycle}.json')
        if counts!=Counter(saved['query_counts']):raise AssertionError('Coverage receipt mismatch')
        for e in es:
            if len(e['query_ids'])!=4 or len(e['context_ids'])!=2 or any(len(c)!=16 or set(c)&set(e['query_ids']) or len(set(c+e['query_ids']))!=20 for c in e['context_ids']):raise AssertionError('Episode shape/identity violation')
        coverage.append(dict(cycle=cycle,updates=len(es),unique_queries=len(counts),query_selections=sum(counts.values())))
    ledger=read(ROOT/'result/rr_v1/gpu_budget.json')
    if any(a.get('seconds') is None for a in ledger['attempts']):raise AssertionError('GPU attempt still open')
    total=ledger['prior_gpu_hours']+sum(a['seconds']/3600 for a in ledger['attempts'])
    if total>ledger['max_gpu_hours']:raise AssertionError('Budget exceeded')
    write(RUN/'postflight.json',dict(status='READY_FOR_DELIVERY_REVIEW',at=now(),actual_updates=30180,coverage=coverage,trace_hashes=trace_hashes,
            final_model_state_hash=last,checkpoint_sha256=latest['sha256'],export_sha256=done['export_sha256'],ledger_sha256=sha(ROOT/'result/rr_v1/gpu_budget.json'),cumulative_gpu_hours=total,contract=frozen,
            boundary='Source training only; no target evaluation, no result claim'))
    print('RR SOURCE POSTFLIGHT READY',flush=True)
if __name__=='__main__':main()
