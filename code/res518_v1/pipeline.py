"""One immutable, metered pipeline from518 source gates to final twelve-target report."""
from supervision import *
TARGET=ROOT/'result/inference/large518_target12_v1'
DATASETS=('mvtec_ad','mpdd','btad','dtd_synthetic','brain_mri','br35h','headct','isic','cvc_clinicdb','cvc_colondb','tn3k','kvasir_seg')

class CompletePipeline(Pipeline):
    def run(self):
        self.step('518_cpu_acceptance_v1',['code/res518_v1/test_resolution.py'])
        self.step('518_target_interface_v1',['code/target518_v1/test_target.py'])
        self.step('518_source_bytes_v1',['code/res518_v1/prepare.py'])
        self.step('518_source_cache_v1',['code/res518_v1/cache_source.py','--mode','extract'],4.)
        self.step('518_cache_postflight_v1',['code/res518_v1/cache_source.py','--mode','audit'])
        self.step('518_gpu_engineering_v1',['code/rr_rank_v1/run.py','--mode','engineering'],2.)
        self.step('518_engineering_diagnostic_postflight_v1',['code/rr_rank_v1/diagnostics.py','--postflight',str(RUN/'engineering/diagnostics')])
        a=read(RUN/'engineering/gpu_acceptance.json')
        if a['contract']!=self.frozen or not a['exact_cuda_recovery'] or a['parameters']!=90126:raise ValueError('Engineering gate failed')
        required=a['conservative_training_gpu_hours']+a['conservative_diagnostic_gpu_hours']+12.+2.
        approved=admission(self.b,required)
        write(RUN/'admission_train.json',dict(status='ADMITTED' if approved else 'BUDGET_INSUFFICIENT',full_updates=30180,
            training_gpu_hours=a['conservative_training_gpu_hours'],diagnostic_gpu_hours=a['conservative_diagnostic_gpu_hours'],
            provisional_target_reserve_gpu_hours=12.,buffer_gpu_hours=2.,total_required_gpu_hours=required,
            available_before_save=self.b['save_at_gpu_hours']-used(self.b),time=now()))
        if not approved:raise RuntimeError('Full training plus reserved inference does not fit; no shortened run')
        self.step('518_train_full_v1',['code/rr_rank_v1/run.py','--mode','train'],a['conservative_training_gpu_hours'])
        self.step('518_training_postflight_v1',['code/rr_rank_v1/postflight.py'])
        self.step('518_final_diagnostics_v1',['code/rr_rank_v1/run.py','--mode','diagnostics'],a['conservative_diagnostic_gpu_hours'])
        self.step('518_final_diagnostic_postflight_v1',['code/rr_rank_v1/diagnostics.py','--postflight',str(RUN/'diagnostics_final')])
        # Bind the actual accepted final export, never an earlier candidate or target-selected checkpoint.
        self.step('518_target_prepare_v1',['code/target518_v1/prepare.py'])
        self.step('518_target_benchmark_v1',['code/target518_v1/worker.py','--mode','benchmark'],1.5)
        bench=read(TARGET/'benchmark.json');required=bench['conservative_total_hours']+1.
        approved=admission(self.b,required)
        write(TARGET/'admission.json',dict(status='ADMITTED' if approved else 'BUDGET_INSUFFICIENT',benchmark_sha256=sha(TARGET/'benchmark.json'),
             conservative_plan_gpu_hours=bench['conservative_total_hours'],buffer_gpu_hours=1.,available_before_save=self.b['save_at_gpu_hours']-used(self.b),time=now()))
        if not approved:raise RuntimeError('Full measured inference does not fit; no target or pass subsampling')
        self.step('518_target_cache_v1',['code/target518_v1/worker.py','--mode','cache'],bench['conservative_cache_hours']+.25)
        self.step('518_target_cache_postflight_v1',['code/target518_v1/worker.py','--mode','cache_audit'])
        stats=read(TARGET/'preflight.json')['datasets'];timings={r['dataset']:r for r in bench['timings']}
        for ds in DATASETS:
            t=timings[ds];limit=2*t['seconds']/t['images']*stats[ds]['images']/3600+.05
            self.step('518_predict_'+ds,['code/target518_v1/worker.py','--mode','predict','--dataset',ds],limit)
            self.step('518_evaluate_'+ds,['code/target518_v1/worker.py','--mode','evaluate','--dataset',ds])
        self.step('518_target_report_v1',['code/target518_v1/worker.py','--mode','report'])
        self.step('518_delivery_evidence_v1',['code/res518_v1/delivery.py'])
        assert all(j['exit_code']==0 for j in self.done) and all(x['seconds'] is not None for x in self.b['attempts'])
        self.stage='ALL_COMPLETE_AWAITING_DELIVERY_REVIEW';self.status('COMPLETE')
        write(RUN/'pipeline_complete.json',dict(status='READY_FOR_DELIVERY_REVIEW',all_children_exit0=True,updates=30180,diagnostic_pairs=144,
             target_datasets=12,raw_maps=31998,fused_maps=10666,cumulative_gpu_hours=used(self.b),ledger_sha256=sha(LEDGER),time=now()))

def main():
    os.chdir(ROOT);manager=None
    with (PARENT/'large_manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:manager=CompletePipeline();manager.run()
        except BaseException as error:
            write(RUN/f'failure_{time.time_ns()}.json',dict(error=repr(error),traceback=traceback.format_exc(),at=now()))
            if manager is not None:manager.status('FAILED',repr(error))
            raise
if __name__=='__main__':main()
