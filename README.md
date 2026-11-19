# ReSetAD

**Learning Reliable References from Image Sets for Anomaly Detection**

[中文说明](README_ZH.md) · [Manuscript](docs/paper.pdf) · [Experiment settings](docs/EXPERIMENT_SETTINGS.md) · [Setup](docs/SETUP.md) · [Results](docs/RESULTS.md)

ReSetAD learns local reference reliability from annotated source image sets and applies it to unlabeled target sets for zero-shot anomaly detection and localization. A frozen DINOv2 encoder supplies visual features. A relation network predicts reference reliability, and reliability penalties adjust matching costs before reference selection. Target inference uses fixed model parameters, without text prompts or verified normal support images.

This repository contains the supplied **DINOv2-L/14, 518 × 518, VisA-source research snapshot**, its learned inference head, archived metrics, and documentation based on the accompanying manuscript and code. The internal implementation name is **RR-Rank**. The manuscript is a submission draft; this repository does not claim conference acceptance.

> **Release scope.** The manuscript evaluates 13 benchmarks, including VisA using a separate MVTec AD-source model. This archive contains the VisA-source model and 12 target datasets only. Some manuscript scores differ from the archived CSV values. See [Results and manuscript comparison](docs/RESULTS.md). Full execution requires external datasets, manifests, caches, and historical execution records.

## Method

1. Extract four frozen feature layers and pool each at three scales, producing 12 branches.
2. Describe cross-image correspondences using distances, ranks, reciprocal matches, and spatial variation.
3. Predict local reference reliability with a shared attention network.
4. Add a reliability penalty before choosing the best anchor per reference image and the two lowest-cost reference images.
5. Convert the averaged deviation into a probability map. Average native-resolution maps over three target groupings, then pool the largest 1% of pixels for the image score.

## Experiment at a glance

| Setting | Released experiment |
|---|---|
| Backbone | Official frozen, non-register DINOv2 ViT-L/14; FP32 |
| Input / feature grid | 518 × 518 / 37 × 37 patches |
| Feature layers | Blocks 6, 12, 18, 24; zero-based indices 5, 11, 17, 23 |
| Branches | Four layers × pooling windows 1, 3, 5 |
| Reference anchors | 18 × 18; odd patch coordinates 1, 3, …, 35 |
| Trainable parameters | 90,126 |
| Source | All 12 VisA categories; 10,821 images: 9,621 normal, 1,200 anomalous |
| Paired training task | Two contexts of 16 images, four shared queries |
| Training | Seed 17; 6,036 reference-pretraining + 24,144 joint updates |
| Optimizer | AdamW; peak LR 2e-4, minimum LR 2e-5; final scheduled checkpoint |
| Matching | Reliability penalty 0.25; two distinct reference images |
| Target grouping | Nominal size 20; seeds 101, 202, 303; query blocks up to four |
| Image score | Top-1% mean after native-resolution probability-map fusion |
| Reported hardware | One NVIDIA A100-SXM4 40 GB (manuscript §4.1) |

See [Experiment settings](docs/EXPERIMENT_SETTINGS.md) for losses, sampling, preprocessing, optimizer groups and evaluation conventions. `configs/paper_settings.json` is a documentation summary, not an executable configuration.

## Get started

```bash
git clone https://github.com/BojueGao/ReSetAD.git
cd ReSetAD
python tools/verify_source_release.py
```

Verification uses Python's standard library only. It checks the **79 archived files** in `provenance/source_release_manifest.json` and parses the Python source; it does not execute training or inference. Added publication files are inventoried in `provenance/publication_manifest.json`.

Follow [Setup and reproduction](docs/SETUP.md) for model use and external dependencies. `requirements-observed.txt` records original environment versions, not a tested fresh-install lockfile. The archived supervisors depend on the original workspace and are not a one-command fresh-checkout training interface.

## Model and results

The released learned head is [`result/rr_rank_v1/inference_model.pt`](result/rr_rank_v1/inference_model.pt), **389,526 bytes**, final update **30,180**, seed **17**.

```text
SHA256 7f46978937a62b06bfc97eccbef2331c5562099ac4c5c79cd39dfdab9b68bcfb
```

The official DINOv2-L/14 implementation and frozen backbone weights are external dependencies. Their provenance is in [`provenance/backbone_weight.json`](provenance/backbone_weight.json).

| Artifact | Location |
|---|---|
| Per-dataset metrics | [metrics_per_dataset.csv](result/inference/large518_target12_v1/metrics/metrics_per_dataset.csv) |
| Per-category metrics | [metrics_per_category.csv](result/inference/large518_target12_v1/metrics/metrics_per_category.csv) |
| Training statistics | [delivery summary](result/delivery_518_target_recovery_v1/) |
| Frozen task plan | [training_plan.json](result/rr_rank_v1/training_plan.json) |
| Source training checks | [postflight.json](result/rr_rank_v1/postflight.json) |

The archive uses one training seed and target-set access. Earlier target diagnostics informed method development. Inference depends on other target images and should be compared under the declared set-based protocol. [Results](docs/RESULTS.md) documents exact archived values and limitations.

## Repository map

```text
code/rr_rank_v1/       Final reliability model, losses, sampler and inference API
code/res518_v1/        Archived 518-resolution preparation and training orchestration
code/target518_v1/     Archived target prediction and evaluation orchestration
code/target14_v1/      Shared target grouping and metric dependencies (legacy name)
src/                  Frozen backbone, feature processing and shared utilities
configs/              Frozen configuration and documentation summary
result/               Learned head, task plan, diagnostics and archived metrics
provenance/           Source hashes, external-asset provenance and publication inventory
docs/                 Manuscript, setup, experiment settings, results and snapshot notes
tools/                Release verification and original job utilities
```

The final RR model is `code/rr_rank_v1/model.py`. Historical architecture and sampler fields in `configs/base.yaml` and `src/models/model.py` do not describe the final RR-Rank model. Preserve the frozen files when checking scientific provenance.

## Licensing and citation

No software license was included in the supplied snapshot, and this preparation does not assign one. Dataset, third-party code and backbone-weight terms remain those of their respective owners. Obtain the relevant permissions before reuse or redistribution. No raw dataset images or masks are included.

Refer to the manuscript title **ReSetAD: Learning Reliable References from Image Sets for Anomaly Detection**. Author names, a public paper identifier and final publication metadata were not supplied, so a bibliographic entry is not fabricated.

See [Release notes](docs/RELEASE_NOTES.md) for preparation changes and [the original snapshot README](docs/SNAPSHOT_README.md) for the supplied archive description.
