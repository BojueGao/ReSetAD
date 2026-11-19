"""Label-blind balanced groups; no borrowed members, cloned images or fixed roles."""
import random,hashlib,json,math
from collections import Counter,defaultdict

def balanced_groups(records,seed,nominal=20):
 if nominal!=20:raise ValueError('Only approved nominal20 main inference')
 if not records:raise ValueError('Empty pool')
 ids=[r['sample_id'] for r in records]
 if len(set(ids))!=len(ids):raise ValueError('Duplicate IDs')
 if len({r['pool_id'] for r in records})!=1:raise ValueError('Mixed domains')
 # Forbidden fields make accidental label/mask leakage fail before sampling.
 if any(any(k in r for k in ('image_label','label','mask_path','anomaly','y')) for r in records):raise ValueError('Label-bearing grouping input')
 n=len(records);k=max(1,math.ceil(n/nominal))
 if n//k<12 and k>1:k-=1
 sizes=[n//k+(i<n%k) for i in range(k)]
 if min(sizes)<12:raise ValueError('Insufficient pool for declared minimum regular group12; no implicit small-set method')
 rng=random.Random(seed);buckets=defaultdict(list)
 for r in sorted(records,key=lambda x:x['sample_id']):buckets[r.get('decoded_rgb_sha256') or r['sha256']].append(r)
 if max(map(len,buckets.values()))>k:raise ValueError('Exact duplicate multiplicity exceeds group count; cannot keep every ID without same-group duplication')
 blocks=list(buckets.values());rng.shuffle(blocks);blocks.sort(key=len,reverse=True)
 groups=[[] for _ in sizes];remaining=sizes.copy()
 for block in blocks:
  rng.shuffle(block);order=list(range(k));rng.shuffle(order);order.sort(key=lambda i:remaining[i],reverse=True)
  for r,i in zip(block,order):
   if not remaining[i]:raise ValueError('No feasible balanced duplicate-free assignment')
   groups[i].append(r);remaining[i]-=1
 for g in groups:rng.shuffle(g)
 assert not any(remaining) and Counter(x['sample_id'] for g in groups for x in g)==Counter(ids)
 return groups

def rotating_turns(group):
 # Four active queries per turn, last turn smaller; every group member scored exactly once.
 if len(group)<12 or len(group)>23:raise ValueError('Unsupported balanced group size')
 for start in range(0,len(group),4):
  qs=group[start:start+4];qids={r['sample_id'] for r in qs};reference=[r for r in group if r['sample_id'] not in qids]
  for offset,q in enumerate(qs):
   context=[r for r in reference if (r.get('decoded_rgb_sha256') or r['sha256'])!=(q.get('decoded_rgb_sha256') or q['sha256'])
            and not (q.get('optional_case_id') and r.get('optional_case_id')==q['optional_case_id'])]
   if len(context)<2:raise ValueError('No valid independent references after known-instance exclusion')
   yield dict(context=context,query=[q],group_ids=[r['sample_id'] for r in group],context_size=len(context),group_size=len(group),
              no_external_reference=True,rotation=start//4,query_slot=offset)

def plan(records,seed):
 turns=[]
 for pool in sorted({r['pool_id'] for r in records}):
  for group in balanced_groups([r for r in records if r['pool_id']==pool],seed):turns.extend(rotating_turns(group))
 return turns
