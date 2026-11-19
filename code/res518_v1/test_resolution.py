"""Source engineering: inherited equations plus odd-grid and448 regression oracles."""
import sys,unittest,tempfile,copy,types,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'code/rr_rank_v1')]
from common import *
import torch,numpy as np
from PIL import Image
from unittest.mock import patch
from test_rr import Mathematics
from test_rank import Revision as ParentRevision
from model import RR
from src.models.backbone import FrozenBackbone
from src.models.relations import branches,fixed_anchors
from src.losses import normality_targets,query_losses
from losses import pixel_terms
from diagnostics import query_targets
from src.cache import FeatureCache,cache_header,canonical_hash
from src.data.adapters import image_tensor,source_mask
import budget

def original(name,rel):
    root=Path(read(ROOT/'provenance/frozen_inputs.json')['original_root'])
    spec=importlib.util.spec_from_file_location(name,root/rel);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Revision(ParentRevision):
    def test_ledger_prefix_and_open_attempt_guards(self):
        b=read(LEDGER);prefix=read(RUN/'budget_prefix_snapshot.json')['ledger'];budget.validate(b,prefix)
        bad=copy.deepcopy(b);bad['prior_gpu_hours']+=1
        with self.assertRaises(ValueError):budget.validate(bad,prefix)
        bad=copy.deepcopy(b);bad['max_gpu_hours']+=1
        with self.assertRaises(ValueError):budget.validate(bad,prefix)
        bad=copy.deepcopy(b);bad['attempts'].append(dict(name='open',seconds=None,start_epoch=0))
        with patch.object(budget,'read',side_effect=[bad,{'ledger':prefix}]):
            with self.assertRaises(RuntimeError):budget.load()
        self.assertFalse(budget.admission(b,192.))
        self.assertTrue(budget.admission(b,100.))
    def test_busy_gpu_selection_and_host_guard(self):
        rows='\n'.join(f'{i}, NVIDIA A100-SXM4-40GB,40960,{3000 if i==0 else 0}' for i in range(4))
        with patch.object(budget.socket,'gethostname',return_value='ubuntu-MZF2-AC0-00'),patch.object(budget.subprocess,'check_output',return_value=rows):self.assertEqual(budget.free_device()[1],1)
        with patch.object(budget.socket,'gethostname',return_value='other-host'):
            with self.assertRaises(RuntimeError):budget.free_device()

class Resolution(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(17)
    def test_immutable_science_tasks_and_model_size(self):
        parent_check();self.assertEqual(sum(p.numel() for p in RR().parameters()),90126)
        self.assertEqual(read(RUN/'training_plan.json')['steps'],30180)
    def test_stride2_odd_grid_has324_unique_anchors(self):
        ids=torch.arange(37*37).reshape(1,37*37,1,1)
        a=fixed_anchors(ids,37,37).flatten()
        expected=torch.tensor([y*37+x for y in range(1,37,2) for x in range(1,37,2)])
        self.assertTrue(torch.equal(a,expected));self.assertEqual(len(a.unique()),324);self.assertEqual(a[-1],35*37+35)
    def test_last_query_row_column_kept(self):
        with torch.no_grad():
            m=RR();st=m.prepare_context(torch.randn(3,4,37,37,8));o=m.score_queries(st,torch.randn(1,4,37,37,8),[(518,518)])
        self.assertEqual(st['weights'].shape,(3,324,12));self.assertEqual(o['logits'].shape,(1,37,37))
        self.assertEqual(o['branch_distances'].shape,(1,1369,12));self.assertTrue(torch.isfinite(o['logits'][0,-1]).all())
        self.assertEqual(o['maps'][0].shape,(518,518))
    def test_boundary_mask_and_branch_alignment(self):
        masks=torch.zeros(2,518,518);masks[1,-1,-1]=1
        q=query_targets(masks);c=normality_targets(masks)
        self.assertEqual(c.shape,(2,324,12));torch.testing.assert_close(q.reshape(2,37,37,12)[:,1::2,1::2].reshape(2,324,12),c,rtol=0,atol=0)
        self.assertEqual(c[1,-1,0],1);self.assertEqual(c[1,-1,1],0);self.assertEqual(c[1,-1,2],0)
        self.assertEqual(q.reshape(2,37,37,12)[1,-1,-1,0],0)
    def test_448_bitexact_reference_labels_losses_and_forward(self):
        oldrel=original('original_rel','src/models/relations.py');oldloss=original('original_loss','src/losses.py')
        oldmodel=original('original_model','code/rr_rank_v1/model.py')
        full=torch.randn(2,1024,12,3);torch.testing.assert_close(fixed_anchors(full,32,32),oldrel.fixed_anchors(full,32,32),rtol=0,atol=0)
        masks=torch.zeros(2,448,448);masks[1,13:40,400:448]=1;labels=torch.tensor([0,1]);logits=torch.randn(2,32,32)
        torch.testing.assert_close(normality_targets(masks),oldloss.normality_targets(masks),rtol=0,atol=0)
        for k,v in pixel_terms(logits,masks,labels).items():torch.testing.assert_close(v,oldloss.query_losses(logits,torch.zeros(2),masks,labels)[k],rtol=0,atol=0)
        m=RR();old=oldmodel.RR();old.load_state_dict(m.state_dict());c=torch.randn(3,4,32,32,4);q=torch.randn(1,4,32,32,4)
        with torch.no_grad():a=m.score_queries(m.prepare_context(c),q,[(448,448)]);b=old.score_queries(old.prepare_context(c),q,[(448,448)])
        torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=0);torch.testing.assert_close(a['scores'],b['scores'],rtol=0,atol=0)
    def test_source_518_mask_keeps_tiny_boundary_defect(self):
        with tempfile.TemporaryDirectory() as td:
            im=Path(td)/'image.png';mask=Path(td)/'mask.png'
            Image.fromarray(np.zeros((1000,1000,3),np.uint8)).save(im)
            array=np.zeros((1000,1000),np.uint8);array[-1,-1]=255;Image.fromarray(array).save(mask)
            self.assertEqual(image_tensor(im).shape,(3,518,518));m=source_mask(dict(label=1,mask_path=str(mask)))
            self.assertEqual(m.shape,(518,518));self.assertEqual(m[-1,-1],1);self.assertGreater(m.sum(),0)
    def test_cache_resolution_isolation_and_wrong_shape_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td)/'weight';w.write_bytes(b'fixture')
            cfg=dict(model=dict(backbone='dinov2_vitl14',layer_indices_zero_based=[5,11,17,23]),execution=dict(backbone_repo=td,backbone_weights=str(w)),preprocess=dict(size=[448,448]),cache=dict(feature_dtype='float32'))
            with patch('src.cache.subprocess.check_output',return_value='commit\n'):
                a=cache_header(cfg);cfg['preprocess']['size']=[518,518];b=cache_header(cfg)
            self.assertEqual(a['shape'],[4,32,32,1024]);self.assertEqual(b['shape'],[4,37,37,1024]);self.assertNotEqual(canonical_hash(a),canonical_hash(b))
            cache=FeatureCache(td,b);r={'image_sha256':'fixture'}
            torch.save(dict(metadata={**b,'sample_sha':'fixture'},features=torch.zeros(4,32,32,1024)),cache.path(r))
            with self.assertRaises(ValueError):cache.read(r)
    def test_dynamic_frozen_backbone_shapes_and_layers(self):
        class Encoder(torch.nn.Module):
            def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.ones(1));self.calls=[]
            def get_intermediate_layers(self,x,**kw):self.calls.append(kw);return [torch.zeros(len(x),(x.shape[2]//14)*(x.shape[3]//14),1024) for _ in range(4)]
        e=Encoder();hub=types.ModuleType('dinov2.hub');hub.backbones=types.SimpleNamespace(dinov2_vitl14=lambda pretrained:e)
        with patch.dict(sys.modules,{'dinov2':types.ModuleType('dinov2'),'dinov2.hub':hub}),patch('torch.load',return_value={'dummy':torch.ones(1)}):
            m=FrozenBackbone('/tmp','fixture','dinov2_vitl14',[5,11,17,23]);m.train()
            for n,g in [(448,32),(518,37)]:self.assertEqual(m(torch.zeros(1,3,n,n)).shape,(1,4,g,g,1024))
            self.assertFalse(m.training);self.assertFalse(any(p.requires_grad for p in m.parameters()))
            self.assertEqual(e.calls[0]['n'],[5,11,17,23]);self.assertFalse(e.calls[0]['return_class_token'])
    def test_odd_grid_permutation_and_query_independence(self):
        m=RR();c=torch.randn(4,4,5,5,8);q=torch.randn(2,4,5,5,8);perm=torch.tensor([2,0,3,1])
        with torch.no_grad():a=m.score_queries(m.prepare_context(c),q);b=m.score_queries(m.prepare_context(c[perm]),q);one=m.score_queries(m.prepare_context(c),q[:1])
        torch.testing.assert_close(a['logits'],b['logits'],atol=1e-6,rtol=1e-6);torch.testing.assert_close(a['logits'][:1],one['logits'],atol=1e-6,rtol=1e-6)
    def test_budget_preserves_old448_intervals_and_new96(self):
        original_budget=read(ROOT/'provenance/large448_budget_snapshot.json');b=read(LEDGER)
        self.assertEqual(b['attempts'][:len(original_budget['attempts'])],original_budget['attempts'])
        self.assertEqual(b['max_gpu_hours']-original_budget['max_gpu_hours'],96.)
        fake=copy.deepcopy(b);fake['attempts']=[dict(seconds=3600),dict(seconds=None,start_epoch=100)]
        self.assertAlmostEqual(budget.used(fake,3700)-fake['prior_gpu_hours'],2)

if __name__=='__main__':
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Mathematics,Revision,Resolution))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    write(RUN/'engineering/cpu_acceptance_v2.json',dict(status='ready',tests=result.testsRun,at=now(),contract=contract(),
         note='Original mathematics, full30180tasks and exact CPUrecovery;518odd-grid,448bitexact,cacheisolation and additional96ledger guards'))
