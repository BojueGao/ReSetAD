# DINOv2-Large 518 — preregistered resolution experiment

Authorized2026-10-30: implement518, full source training, automatic target inference/evaluation, add96GPUh.
Compared with completed448 Large, only image size and derived grids change. No target-score-based tuning.
VisA all12/10821 source images; original source bytes, neighbors,30180 task identities, eta and144diagnostic identities.
Seed17 new initialization of90126learnable parameters; frozen official non-register DINOv2-L/14 FP32 layers[5,11,17,23].
6036warmup +4*6036joint=30180updates; joint horizon24144. Keep original optimizer/loss weights and paired C16/Q4 tasks.
Input518x518 PIL bicubic, sameImageNetnormalization;37x37query patches; anchors indices1,3,...,35 =>18x18.
Keep all query patches including row/column36. No padding, duplicate samples, cropping or extra attention mechanism.
Reference supervision uses exactly same anchor indices and r1/3/5 max-supported source-mask footprints.
Source masks retain original area-positive-support resampling; evaluation masks keep audited native binary masks.
Target decoder retains approvedS002 RGBA alpha discard and palette expansion, only resize changes to518.
Three seed partitions101/202/303, balanced nominal20sets, unchanged query rotation and duplicate/case exclusion.
Native FP32 probability maps averaged before top1%image score. No target updates, model selection, threshold tuning or group normalization.
12targets: MVTecAD,MPDD,BTAD,DTD-Synthetic,BrainMRI,Br35H,HeadCT,ISIC2016,CVC-ClinicDB,CVC-ColonDB,TN3K,Kvasir-SEG.
Real-IAD/KSDD excluded. VisA is source, no self-test or reverse-source model implicit.
Metrics frozen: imageAUROC/standardAP; exactall-FP32pixelAUROC; normalizedAUPRO@.30,8connected,4096globalthresholds.
Single-label image metrics and absent real pixel masks remainNA. Original-size map geometry and all hashes audited.

Pipeline: CPU/source gates -> source cache -> cacheCPUaudit ->24warmup/200joint GPUbenchmark and exactCUDArecovery ->
12engineering diagnosticpairs and CPUaudit -> full-budgetadmission -> fulltraining -> sourcepostflight ->144fixeddiagnostics/CPUaudit ->
bind final30180export only -> targetbenchmark -> targetbudgetadmission -> targetcache/audit ->12predictions/fusion/metrics -> terminalreport/package.
Training checkpoints include model/optimizer/scheduler/RNG/sampler. No implicit retry or shortening to fit budget.
Before training reserve measured conservative fulltraining+diagnostics and12GPUh for target stage plus2GPUhbuffer.
Before target use actual targetbenchmark to reserve complete remaining plan plus1GPUhbuffer. Failure stops with original evidence.
One sequential GPU child and CPU-only audit/evaluation stages; CPU work never chargedGPUh. GPU interval includes startup and process exit.

Budget: retain historic91.70778896749681GPUh plus all completed448Large attempts (~36.59406635483106GPUh).
User adds96 to existing96 allowance: allowance192, cumulative hardcap283.7077889674968, save283.4577889674968GPUh.
Expected available at creation155.40593364516894h; actual verified ledger controls. Old ledger frozen read-only, new ledger includes original attempts unchanged.
Benchmark estimates are not results. Single seed and prior target-informed design history remain disclosed.
