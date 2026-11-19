"""New independent fixtures: readout semantics, gradient ownership, eta, purity."""
from common import *
import unittest,math
import numpy as np
import torch
from losses import evidence_from_distance,rank_evidence,pixel_terms,normalized_rank,paired_step
from diagnostics import scalar_auc,summarize_arrays,selected_arrays,query_targets,context_gaps
from model import RR
from src.losses import query_losses,normality_targets
from src.engine import seed_all,optimizer_for,StageScheduler,state_hash

class Revision(unittest.TestCase):
    def setUp(self):seed_all(17);torch.set_num_threads(1)
    def test_constant_energy_scalar_oracle(self):
        d=torch.tensor([[[.2]],[[.8]]],requires_grad=True);rho=torch.tensor(.7,requires_grad=True);b=torch.tensor(-1.5,requires_grad=True)
        e,s=evidence_from_distance(d,[(7,5),(5,9)],rho,b)
        torch.testing.assert_close(e,torch.tensor([.2,.8]),rtol=1e-6,atol=1e-6)
        e.sum().backward();self.assertIsNone(rho.grad);self.assertIsNone(b.grad);torch.testing.assert_close(d.grad,torch.ones_like(d),rtol=1e-6,atol=1e-6)
    def test_ranking_monotone_and_exact_score(self):
        m=RR();st=m.prepare_context(torch.randn(3,4,4,4,8));o=m.score_queries(st,torch.randn(4,4,4,4,8),[(17,19)]*4)
        e=rank_evidence(m,o,[(17,19)]*4)
        self.assertTrue(torch.equal(torch.argsort(e),torch.argsort(o['scores'])))
    def test_rank_grad_only_reliability(self):
        m=RR();m.set_stage('joint');st=m.prepare_context(torch.randn(3,4,4,4,8));st['weights'].retain_grad()
        o=m.score_queries(st,torch.randn(2,4,4,4,8),[(17,19)]*2);e=rank_evidence(m,o,[(17,19)]*2)
        # Always optimize the lower-scored sample as positive, forcing active hinge.
        labels=torch.zeros(2,dtype=torch.long);labels[int(e.argmin())]=1
        rank,_,n=normalized_rank(torch.stack([e,e]),labels,.4);rank.backward()
        self.assertEqual(n,1);self.assertIsNone(m.rho.grad);self.assertIsNone(m.bias.grad)
        self.assertGreater(st['weights'].grad.abs().sum().item(),0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in m.reliability.parameters() if p.grad is not None),0)
    def test_pixel_exact_parent(self):
        logits=torch.randn(2,32,32,requires_grad=True);masks=torch.zeros(2,448,448);masks[1,40:80,90:100]=1;labels=torch.tensor([0,1])
        actual=pixel_terms(logits,masks,labels);old=query_losses(logits,torch.tensor([-1.,1.]),masks,labels)
        for key in actual:torch.testing.assert_close(actual[key],old[key],rtol=0,atol=0)
        actual['pixel'].backward();self.assertGreater(logits.grad.abs().sum().item(),0)
    def test_empty_rank_differentiable(self):
        e=torch.ones(2,4,requires_grad=True);rank,raw,n=normalized_rank(e,torch.zeros(4,dtype=torch.long),.25)
        self.assertEqual(n,0);self.assertEqual(rank.item(),0);rank.backward();self.assertEqual(e.grad.abs().sum().item(),0)
    def test_eta_hand_mean(self):
        # One eligible of four same-class tasks: conditional mean remains .3.
        e=torch.tensor([[.4,.2],[.4,.2]]);rank,raw,n=normalized_rank(e,torch.tensor([0,1]),.25)
        self.assertAlmostEqual(raw.item(),.3,places=6);self.assertAlmostEqual(rank.item()/4,.3,places=6)
        for bad in (0.,-1.,1.1,float('nan')):
            with self.assertRaises(ValueError):normalized_rank(e,torch.tensor([0,1]),bad)
    def test_numerical_guards_no_silent_clipping(self):
        d=torch.ones(1,1,1)
        for rho,bias in [(-30.,0.),(1.,50.),(1.,-50.)]:
            with self.assertRaises(FloatingPointError):evidence_from_distance(d,[(3,3)],torch.tensor(rho),torch.tensor(bias))
    def test_auc_hand_ties(self):
        self.assertAlmostEqual(scalar_auc([.1,.5,.5,.9],[0,0,1,1]),.875)
        self.assertIsNone(scalar_auc([.2,.3],[1,1]))
    def test_selected_pollution_scalar_fixture(self):
        # Two references, two anchors, one branch: selected bad then good => .5.
        w=np.array([[[.9],[.1]],[[.2],[.8]]],np.float32);cn=np.array([[[1],[0]],[[0],[1]]])
        matches=np.array([[[[1,1]]]]);refs=np.array([[[[0,1]]]]);qn=np.ones((1,1,1))
        r=summarize_arrays(w,matches,refs,cn,qn)
        self.assertEqual(r['selection']['all']['anomaly_fraction'],.5)
        self.assertEqual(r['selection']['all']['count'],2);self.assertIsNone(r['selection']['query_anomaly_patch']['anomaly_fraction'])
        self.assertEqual(r['high_weight_normal_purity'],1.)
        self.assertAlmostEqual(r['reference_anomaly_mass'],.15,places=6)
        bad=refs.copy();bad[...,1]=0
        with self.assertRaises(ValueError):selected_arrays(w,matches,bad)
    def test_zero_mass_and_mask_branch_alignment(self):
        masks=torch.zeros(2,448,448);masks[1,14:28,14:28]=1
        q=query_targets(masks);c=normality_targets(masks)
        torch.testing.assert_close(q.reshape(2,32,32,12)[:,1::2,1::2].reshape(2,256,12),c,rtol=0,atol=0)
        w=np.zeros((2,2,1),np.float32);cn=np.ones_like(w);m=np.zeros((1,1,1,2),np.int64);r=np.array([[[[0,1]]]])
        summary=summarize_arrays(w,m,r,cn,np.ones((1,1,1)))
        self.assertIsNone(summary['reference_anomaly_mass']);self.assertIsNone(summary['high_weight_normal_purity'])
    def test_frozen_plan_eta_source_only(self):
        plan=read(RUN/'training_plan.json');self.assertEqual(len(plan['task_digests']),30180)
        self.assertEqual(sum(plan['joint_tasks'].values()),24144)
        for c,eta in plan['eta'].items():self.assertEqual(eta,plan['mixed_tasks'][c]/plan['joint_tasks'][c])
        manifest=read(RUN/'diagnostic_manifest.json');self.assertEqual(len(manifest['tasks']),144)
        self.assertEqual(len({t['id'] for t in manifest['tasks']}),144)
    def test_context_gap_hand_oracle(self):
        records=[dict(task_id='x',variant='learned',view=v,query_ids=['n','a'],query_labels=[0,1],scores=s,category='fixture',k=8,style='random') for v,s in enumerate(([.2,.8],[.3,.7]))]
        r=context_gaps(records,1.,0.)[0]
        self.assertAlmostEqual(r['mean_absolute_score_shift'],.1)
        self.assertAlmostEqual(r['worst_pair_score_gap'],.4)
        self.assertAlmostEqual(r['worst_pair_E_gap'],2*math.log(.7/.3))
        for r in records:r['query_labels']=[0,0]
        self.assertIsNone(context_gaps(records,1.,0.)[0]['worst_pair_E_gap'])
    def test_ledger_prefix_and_open_attempt_guards(self):
        import supervise
        from unittest.mock import patch
        from copy import deepcopy
        prefix=read(RUN/'budget_prefix_snapshot.json')['ledger'];bad=deepcopy(prefix);bad['attempts'][0]['seconds']+=1
        with patch.object(supervise,'read',side_effect=[{'ledger':prefix},bad]):
            with self.assertRaises(ValueError):supervise.load_ledger()
        bad=deepcopy(prefix);bad['attempts'].append(dict(name='open',seconds=None,start_epoch=0))
        with patch.object(supervise,'read',side_effect=[{'ledger':prefix},bad]):
            with self.assertRaises(RuntimeError):supervise.load_ledger()
        with patch.object(supervise,'device_check',side_effect=AssertionError('Must reject before GPU allocation')):
            with self.assertRaises(RuntimeError):supervise.allocate(prefix,'fixture','train',10.)
    def test_busy_gpu_selection_and_host_guard(self):
        import supervise
        from unittest.mock import patch
        rows='\n'.join(f'{i}, NVIDIA A100-SXM4-40GB, 40960, {3045 if i==0 else 0}' for i in range(4))
        with patch.object(supervise.subprocess,'check_output',side_effect=['ubuntu-MZF2-AC0-00',rows]):
            devices,gpu=supervise.device_check();self.assertEqual(gpu,1);self.assertEqual(len(devices),4)
        busy='\n'.join(f'{i}, NVIDIA A100-SXM4-40GB, 40960, 3045' for i in range(4))
        with patch.object(supervise.subprocess,'check_output',side_effect=['ubuntu-MZF2-AC0-00',busy]):
            with self.assertRaises(RuntimeError):supervise.device_check()
        with patch.object(supervise.subprocess,'check_output',return_value='wrong-host'):
            with self.assertRaises(RuntimeError):supervise.device_check()

if __name__=='__main__':
    import test_rr
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(test_rr.Mathematics),unittest.defaultTestLoader.loadTestsFromTestCase(Revision)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    write(RUN/'engineering/cpu_acceptance_v2.json',dict(status='ready',tests=result.testsRun,at=now(),contract=contract(),revision_reason='Free A100 selection; prior CPU acceptance and pre-allocation rejection preserved'))
