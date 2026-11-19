"""CPU-only three-round target fixture, S002 decode, final binding and NA semantics."""
import unittest,tempfile,json,copy
from pathlib import Path
from unittest.mock import patch
import worker as w
import torch,numpy as np
from PIL import Image
from target.data import read_rgb,image_tensor
from evaluate import image_metrics,evaluate
import evaluate as target_eval

class Target(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(17)
    def test_s002_palette_and_alpha_518(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'rgba.png';x=np.zeros((7,9,4),np.uint8);x[:,:,:3]=[230,10,50];Image.fromarray(x).save(p)
            self.assertEqual(np.asarray(read_rgb(p))[0,0].tolist(),[230,10,50]);self.assertEqual(image_tensor(p).shape,(3,518,518))
            image=Image.new('P',(4,4));palette=[0]*768;palette[0:3]=[10,20,30];image.putpalette(palette);image.save(p)
            self.assertEqual(np.asarray(read_rgb(p))[0,0].tolist(),[10,20,30])
    def test_single_label_image_metrics_are_null(self):
        for labels in ([0,0],[1,1]):
            r=image_metrics(labels,[.2,.8]);self.assertIsNone(r['image_auroc']);self.assertIsNone(r['image_ap'])
        r=image_metrics([0,1],[.2,.8]);self.assertEqual(r['image_auroc'],1);self.assertEqual(r['image_ap'],1)
    def test_complete_native_three_round_fusion_and_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ds='fixture';run=root/'inference';rows=[dict(sample_id=str(i),sha256=str(i),pool_id=ds,height=12+i%3,width=14) for i in range(20)]
            features={r['sample_id']:torch.randn(4,5,5,8) for r in rows};provider=lambda rr:torch.stack([features[r['sample_id']] for r in rr])
            m=w.rrmodel.RR().requires_grad_(False).eval();h=w.state_hash(m);pred=run/'predictions'/ds
            ann=[]
            for i,r in enumerate(rows):
                mask=None;mh=None
                if i>=10:
                    mask=root/f'mask{i}.png';v=np.zeros((r['height'],r['width']),np.uint8);v[2:5,3:6]=255;Image.fromarray(v).save(mask);mh=w.sha(mask)
                ann.append(dict(image_id=r['sample_id'],category=ds,image_label=int(i>=10),mask_path=str(mask) if mask else None,mask_sha256=mh,annotation_kind='binary_anomaly_mask',height=r['height'],width=r['width']))
            auditroot=root/'dataset_audit'/ds;w.write(auditroot/'manifest.json',ann);w.write(auditroot/'audit.json',dict(status='ready',manifest_sha256=w.sha(auditroot/'manifest.json')))
            with patch.object(w,'RUN',run),patch.object(w,'STATE_SHA',h),patch.object(w,'records',return_value=rows),patch.object(target_eval,'RESULT',root):
                w.predict_raw(m,rows,provider,pred);w.fuse_audit(ds);result=evaluate(ds,pred/'fused',run/'metrics'/ds)
                self.assertEqual(result[0]['images'],20);self.assertEqual(w.state_hash(m),h)
                for k in ('image_auroc','image_ap','pixel_auroc','aupro_030'):self.assertTrue(0<=result[0][k]<=1)
                post=w.read(run/'postflight'/f'{ds}.json');self.assertEqual(post['raw_maps'],60);self.assertEqual(post['fused_maps'],20)
                fused=w.read(pred/'fused/predictions.json');self.assertTrue(all(x['pass_count']==3 for x in fused))
    def test_target_pipeline_order_is_frozen(self):
        text=(w.ROOT/'code/res518_v1/pipeline.py').read_text()
        self.assertLess(text.index("'518_train_full_v1'"),text.index("'518_training_postflight_v1'"))
        self.assertLess(text.index("'518_final_diagnostic_postflight_v1'"),text.index("'518_target_prepare_v1'"))
        self.assertLess(text.index("'518_target_benchmark_v1'"),text.index("'518_target_cache_v1'"))
        self.assertNotIn('source_protocol_lock',text)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Target))
    if not result.wasSuccessful():raise SystemExit(1)
    w.write(w.TRAIN/'engineering/target_interface_acceptance.json',dict(status='ready',tests=result.testsRun,time=w.now(),scientific_prediction_executed=False,fixture_only=True))
