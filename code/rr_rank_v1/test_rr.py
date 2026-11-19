"""Independent numerical fixtures and real-manifest sampler acceptance (CPU)."""
from common import ROOT,RUN,read,write,source,now,contract
import math,unittest,tempfile,inspect
import numpy as np
import torch
import torch.nn.functional as F
from model import RR,edge_features,image_score,fuse_maps
from losses import pair_rank
from inference import balanced_groups,rotating_turns,predict_group
from sampling import PairSampler
from src.models.relations import imagewise_min
from src.engine import seed_all,optimizer_for,StageScheduler,save_checkpoint,restore_checkpoint,state_hash


def slow_edges(x,h,w):
    c,a,b,d=x.shape;result=np.empty((c,a,c-1,b,7))
    for i in range(c):
        for branch in range(b):
            matrices={j:np.clip(1-x[i,:,branch]@x[j,:,branch].T,0,2) for j in range(c) if j!=i}
            for u in range(a):
                other=list(matrices);ds=[float(matrices[j][u].min()) for j in other]
                y,z=divmod(u,w);adj=[yy*w+xx for yy,xx in ((y-1,z),(y+1,z),(y,z-1),(y,z+1)) if 0<=yy<h and 0<=xx<w]
                for index,j in enumerate(other):
                    mat=matrices[j];order=np.sort(mat[u]);v=int(np.argmin(mat[u]));reverse=int(np.argmin(mat[:,v]))
                    rank=(sum(t<ds[index] for t in ds)+.5*(sum(t==ds[index] for t in ds)-1))/max(1,c-2)
                    result[i,u,index,branch]=[order[0]/2,order[:4].mean()/2,(order[1]-order[0])/2,rank,float(reverse==u),np.mean([abs(ds[index]-mat[t].min()) for t in adj])/2,math.log(c)/math.log(32)]
    return result.reshape(c*a,c-1,84)


class Mathematics(unittest.TestCase):
    def setUp(self):seed_all(17);torch.set_num_threads(1)
    def test_edge_oracle(self):
        a=F.normalize(torch.randn(4,16,12,7,dtype=torch.float64),dim=-1)
        actual=edge_features(a,4,4).numpy();expected=slow_edges(a.numpy(),4,4)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    def test_rank_ties_and_self_exclusion(self):
        a=torch.ones(4,4,12,2)/math.sqrt(2)
        edges=edge_features(a,2,2).reshape(4,4,3,12,7)
        self.assertTrue(torch.all(edges[...,3]==.5));self.assertEqual(edges.shape[2],3)
    def test_penalty_precedes_min(self):
        nodes=torch.tensor([[[[1.,0.]]]])
        anchors=torch.tensor([[[[1.,0.]],[[.9,math.sqrt(.19)]]],[[[.8,.6]],[[0.,1.]]]])
        weights=torch.tensor([[[0.],[1.]],[[1.],[1.]]],requires_grad=True)
        values,indices=imagewise_min(nodes,anchors,weights,torch.tensor([.25]))
        self.assertEqual(indices[0,0,0,0].item(),1)
        self.assertAlmostEqual(values[0,0,0,0].item(),.1,places=6)
        self.assertAlmostEqual(values.topk(2,dim=-1,largest=False).values.mean().item(),.15,places=6)
        values.sum().backward();self.assertEqual(weights.grad[0,0,0].item(),0);self.assertAlmostEqual(weights.grad[0,1,0].item(),-.25)
    def test_uniform_weights_raw_reference(self):
        x=F.normalize(torch.randn(2,5,12,7),dim=-1);a=F.normalize(torch.randn(3,4,12,7),dim=-1)
        weighted,_=imagewise_min(x,a,torch.ones(3,4,12),torch.full((12,),.25))
        raw,_=imagewise_min(x,a);torch.testing.assert_close(weighted,raw,rtol=0,atol=0)
    def test_context_permutation_and_query_independence(self):
        model=RR().eval();c=torch.randn(4,4,4,4,8);q=torch.randn(2,4,4,4,8)
        s=model.prepare_context(c);original=model.score_queries(s,q,[(16,16)]*2)
        perm=torch.tensor([2,0,3,1]);ps=model.prepare_context(c[perm]);out=model.score_queries(ps,q,[(16,16)]*2)
        torch.testing.assert_close(s['weights'][perm],ps['weights'],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(original['logits'],out['logits'],atol=1e-6,rtol=1e-6)
        one=model.score_queries(s,q[:1],[(16,16)]);torch.testing.assert_close(one['logits'],original['logits'][:1],atol=1e-6,rtol=1e-6)
        self.assertNotIn('label',inspect.signature(model.prepare_context).parameters)
    def test_downstream_reference_gradient(self):
        model=RR();s=model.prepare_context(torch.randn(3,4,4,4,8));s['weights'].retain_grad()
        out=model.score_queries(s,torch.randn(1,4,4,4,8),[(16,16)]);out['scores'].sum().backward()
        self.assertGreater(s['weights'].grad.abs().sum().item(),0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.reliability.parameters() if p.grad is not None),0)
        self.assertGreater(F.softplus(model.rho).item(),0)
    def test_rank_empty_and_worst_case(self):
        s=torch.tensor([[.3,.4],[.2,.6]],requires_grad=True)
        loss,n=pair_rank(s,torch.tensor([0,1]));self.assertEqual(n,1);self.assertAlmostEqual(loss.item(),0.,places=6)
        loss,n=pair_rank(s,torch.tensor([1,0]));self.assertAlmostEqual(loss.item(),.5,places=6)
        empty,n=pair_rank(s,torch.tensor([0,0]));self.assertEqual(n,0);self.assertEqual(empty.item(),0);empty.backward()
    def test_pixel_fusion_before_readout(self):
        maps=[torch.zeros(10,10) for _ in range(3)]
        for i,m in enumerate(maps):m[i,0]=.9
        fused,score=fuse_maps(maps);self.assertAlmostEqual(score.item(),.3,places=6)
        self.assertAlmostEqual(torch.stack([image_score(m) for m in maps]).mean().item(),.9,places=6)
    def test_tail_duplicate_and_three_round_coverage(self):
        for count in (12,20,21,23,24,33,41,101):
            rows=[dict(sample_id=str(i),sha256=str(i),pool_id='fixture') for i in range(count)]
            counts={r['sample_id']:0 for r in rows}
            for seed in (101,202,303):
                groups=balanced_groups(rows,seed)
                self.assertLessEqual(max(map(len,groups))-min(map(len,groups)),1)
                for g in groups:
                    for t in rotating_turns(g):
                        self.assertEqual(len(t['query']),1);counts[t['query'][0]['sample_id']]+=1
                        self.assertFalse({r['sample_id'] for r in t['context']}&{r['sample_id'] for r in t['query']})
            self.assertEqual(set(counts.values()),{3})
        rows=[dict(sample_id=str(i),sha256=str(i//2),pool_id='fixture') for i in range(40)]
        groups=balanced_groups(rows,101)
        self.assertTrue(all(len({r['sha256'] for r in g})==len(g) for g in groups))
        with self.assertRaises(ValueError):balanced_groups([{**r,'label':0} for r in rows],101)
    def test_whole_group_api_native_shape_state_and_coverage(self):
        rows=[dict(sample_id=str(i),sha256=str(i),pool_id='fixture',height=10+i,width=16) for i in range(12)]
        features={r['sample_id']:torch.randn(4,4,4,8) for r in rows}
        provider=lambda rs:torch.stack([features[r['sample_id']] for r in rs])
        m=RR();before=state_hash(m);out=predict_group(m,rows,provider)
        self.assertEqual(before,state_hash(m));self.assertEqual(len(out),12)
        for r in rows:self.assertEqual(tuple(out[r['sample_id']].shape),(r['height'],r['width']))
    def test_warmup_readout_freeze_and_native_image_gradient(self):
        m=RR();m.set_stage('warmup');self.assertFalse(m.rho.requires_grad);self.assertFalse(m.bias.requires_grad)
        m.set_stage('joint');s=m.prepare_context(torch.randn(3,4,4,4,8))
        out=m.score_queries(s,torch.randn(2,4,4,4,8),[(33,27),(17,14)])
        loss=F.binary_cross_entropy(out['scores'],torch.tensor([0.,1.]));loss.backward()
        self.assertTrue(torch.isfinite(m.rho.grad));self.assertGreater(sum(p.grad.abs().sum().item() for p in m.reliability.parameters() if p.grad is not None),0)
    def test_actual_source_five_cycles_and_sampler_restore(self):
        rows=source();neighbors=read(ROOT/'result/rr_v1/source_neighbors.json');sampler=PairSampler(rows,neighbors)
        reports=[];ratios={};rank_empty=0
        for cycle in range(5):
            sampler.begin_cycle('warmup' if cycle==0 else 'joint')
            for i in range(6036):
                before=sampler.state_dict() if i==13 else None
                e=sampler.next();self.assertTrue(all(r['label']==0 for r in e['contexts'][0]))
                self.assertEqual(sum(r['label'] for r in e['contexts'][1]),e['k'])
                if i==13:
                    clone=PairSampler(rows,neighbors);clone.load_state_dict(before);self.assertEqual(e,clone.next())
                ratios[e['k']]=ratios.get(e['k'],0)+1
                rank_empty+=len({r['label'] for r in e['query']})==1
            reports.append(sampler.audit_coverage())
        write(RUN/'engineering/sampler_five_cycles_v2.json',dict(status='ready',cycles=reports,k_counts=ratios,rank_empty_tasks=rank_empty,total_tasks=30180))
    def test_optimizer_rng_exact_recovery(self):
        rows=source();neighbors=read(ROOT/'result/rr_v1/source_neighbors.json');sampler=PairSampler(rows,neighbors);sampler.begin_cycle('joint')
        model=RR();opt=optimizer_for(model);sched=StageScheduler(opt,24144)
        def step():
            sampler.next();opt.zero_grad();sched.prepare();loss=sum((p*torch.randn_like(p)).square().mean() for p in model.parameters());loss.backward();opt.step();sched.advance();return float(loss.detach())
        step()
        with tempfile.TemporaryDirectory() as td:
            p=td+'/exact.pt';save_checkpoint(p,model,opt,sched,sampler,{'fixture':1})
            expected=step();expectedhash=state_hash(model);expectedstate=sampler.state_dict()
            restore_checkpoint(p,model,opt,sched,sampler,{'fixture':1});actual=step()
            self.assertEqual(actual,expected);self.assertEqual(state_hash(model),expectedhash);self.assertEqual(sampler.state_dict(),expectedstate)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Mathematics))
    if not result.wasSuccessful():raise SystemExit(1)
    write(RUN/'engineering/cpu_acceptance.json',dict(status='ready',tests=result.testsRun,at=now(),contract=contract()))
