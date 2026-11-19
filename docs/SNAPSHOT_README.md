# ReSetAD

Research code for **Learning Reliable References from Image Sets for Anomaly Detection**. This snapshot contains the completed DINOv2-L/14 **518 × 518** experiment (internal implementation name: RR-Rank), retrieved from A100-15 on 2026-11-18.

## Method and protocol

- Frozen, non-register DINOv2-L/14; intermediate layers 6, 12, 18, 24; 12 spatial branches.
- 90,126 trainable parameters for reference reliability and the shared readout.
- Source: all 12 VisA categories, 10,821 images, with source labels and masks. The original source train/test partitions are merged.
- Training: seed 17, 6,036 reference-pretraining updates and 24,144 joint updates; final scheduled checkpoint at update 30,180.
- Paired tasks use four shared queries and 16 context images per view. Queries follow the mixed-label queue, without a fixed normal/anomalous balance.
- Target: nominal 20-image, label-free balanced sets; query rotation; grouping seeds 101, 202, 303. Three native-resolution probability maps are averaged pixel-wise before computing the top-1% image score.
- Twelve evaluated target datasets; Real-IAD, KSDD and source-domain VisA self-testing are excluded.

The effective protocol is defined by `configs/resolution518.json`, `configs/rr_rank_v1.json`, the frozen task plan, and `docs/RESOLUTION_518_PROTOCOL_V1.md`. `configs/base.yaml` retains historical development/sampler fields: those fields do **not** describe the final RR-Rank sampling protocol.

## Code map

| Path | Purpose |
|---|---|
| `code/rr_rank_v1/model.py` | Relation reliability, reference matching, shared readout |
| `code/rr_rank_v1/sampling.py` | Source task sampling |
| `code/rr_rank_v1/losses.py` | Reference, pixel, and ranking objectives |
| `code/rr_rank_v1/run.py` | Training and checkpoint export |
| `code/rr_rank_v1/inference.py` | Model inference |
| `code/rr_rank_v1/diagnostics.py` | Fixed source diagnostic implementation |
| `code/res518_v1/` | Resolution-specific preparation, cache, engineering and orchestration |
| `code/target518_v1/` | Target prediction and evaluation orchestration |
| `code/target14_v1/` | Shared grouping, data and metric dependencies; legacy name retained |
| `src/` | Frozen backbone, feature processing and shared utilities |
| `tools/` | Original job/receipt utilities |

## Model and results

The learned inference export is included at `result/rr_rank_v1/inference_model.pt` (389,526 bytes):

```text
SHA256 7f46978937a62b06bfc97eccbef2331c5562099ac4c5c79cd39dfdab9b68bcfb
```

It requires the separate official frozen DINOv2-L/14 backbone. The original source URL, SHA256 and server location are recorded in `provenance/backbone_weight.json`. The backbone and the upstream DINOv2 implementation are not redistributed here.

- Dataset metrics: `result/inference/large518_target12_v1/metrics/metrics_per_dataset.csv`
- Category metrics: `result/inference/large518_target12_v1/metrics/metrics_per_category.csv`
- Training statistics: `result/delivery_518_target_recovery_v1/`
- Source postflight: `result/rr_rank_v1/postflight.json`

Metrics include image AUROC, standard AP, pixel AUROC and normalized AUPRO@0.30. Empty entries represent inapplicable metrics. The results use one training seed and target-set access; they should not be described as strictly independent single-image inference. Earlier target diagnostics informed method development.

## Reproducibility and external assets

This is a byte-preserving research snapshot, not a standalone offline runtime. The archived pipeline checks original paths, data manifests, feature caches, historical contracts and budget records. Those checks are preserved; running the original supervisors in a fresh checkout without these dependencies is not supported.

The original executable workspace is `/home/ubuntu/MyDisk/VSI-AD-Large-518`, using `/home/ubuntu/MyDisk/VSI-AD/.venv/bin/python`. Its completed jobs must not be restarted to inspect this release.

External assets required to reproduce execution include:

1. Licensed source/target datasets and their audited manifests; no raw images or masks are distributed.
2. Official DINOv2 code and backbone weights.
3. Feature caches, original input manifests and lifecycle/contract records referenced by the archived code.
4. The original environment and a separate experiment/output directory with newly established execution contracts.

`requirements-observed.txt` records the original core environment versions; it is not a tested fresh-install lockfile. The full optimizer checkpoint remains on A100-15 at `result/rr_rank_v1/checkpoints/step_030180.pt`, SHA256 `e31d11ea96aa337a0ce65c81c512f74fbab8e4b104f85d585e14839eaed71e0a`.

## Verify this snapshot

No GPU or model execution is needed:

```bash
python tools/verify_source_release.py
```

This verifies every copied file against the source SHA256 manifest and parses the Python source. It does not repeat training, target inference or scientific acceptance.

## Licensing

No new software license is assigned by this upload. Dataset and third-party software/weight licenses remain those of their respective owners. Authors must choose an explicit license before representing this repository as permissively licensed software.
