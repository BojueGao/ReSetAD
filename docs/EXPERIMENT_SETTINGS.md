# Experiment settings

This document distinguishes the manuscript protocol from the supplied executable snapshot. Manuscript references point to [the supplied PDF](paper.pdf), especially §§3.3–3.6 and §4.1 (pages 3–5). Implementation details come from the frozen files linked below. No experiment was rerun during repository preparation.

## Configuration authority

| File | Role |
|---|---|
| [resolution518.json](../configs/resolution518.json) | Resolution-specific final experiment |
| [rr_rank_v1.json](../configs/rr_rank_v1.json) | RR-Rank losses and schedule |
| [training_plan.json](../result/rr_rank_v1/training_plan.json) | Frozen 30,180 task digests and category ranking normalization |
| [sampling.py](../code/rr_rank_v1/sampling.py) | Actual paired task sampler |
| [model.py](../code/rr_rank_v1/model.py) | Actual reliability architecture and readout |
| [losses.py](../code/rr_rank_v1/losses.py) | Actual final optimization objective |
| [engine.py](../src/engine.py) | Optimizer groups, learning-rate schedule and random state |
| [original 518 protocol](RESOLUTION_518_PROTOCOL_V1.md) | Historical experiment contract |

`configs/base.yaml` retains development-era entries, including held-out categories, fixed 2+2 query balance, variable context sizes, an older set encoder, three final seeds and source checkpoint selection. Those entries must not be read as final RR-Rank settings. The release uses all 12 source classes, a mixed-label query queue, fixed context size 16, one seed and the final scheduled checkpoint. [paper_settings.json](../configs/paper_settings.json) is a documentation summary; archived launchers do not consume it.

## Data and preprocessing

| Item | Setting |
|---|---|
| Source | VisA, all 12 categories; original train/test partitions merged |
| Source counts | 10,821 images: 9,621 normal and 1,200 anomalous |
| Labels | Image labels and binary masks used only for source supervision and evaluation |
| Categories | candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum |
| Image size | 518 × 518, direct PIL bicubic resize |
| Normalization | RGB / 255; mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] |
| Augmentation | None in the archived preprocessing configuration |
| Source masks | Area resampling followed by positive-support binarization |
| Evaluation masks | Audited native binary masks; original map geometry retained |

The source adapter applies EXIF orientation and converts supported images to RGB. The archived target decoder additionally handles approved palette/RGBA cases; alpha is discarded rather than used as a label. Data are not redistributed. See [source adapters](../src/data/adapters.py) and [target decoding](../code/target518_v1/target/data.py).

## Features and reliability network

| Item | Setting |
|---|---|
| Encoder | Official pretrained DINOv2 ViT-L/14, non-register variant, frozen |
| Feature width | 1,024 per layer |
| Layers | Blocks {6,12,18,24}; zero-based {5,11,17,23} |
| Branches | Four layers × average-pooling windows {1,3,5}; reflect padding; L2 normalization eps 1e-6 |
| Query grid | Full 37 × 37, including last row/column |
| Anchor grid | Odd coordinates 1,3,…,35 in both axes; 18 × 18 |
| Descriptor | 7 statistics per branch × 12 = 84 features per cross-image edge |
| Descriptor statistics | Nearest half-cosine distance, mean of nearest 4, nearest-neighbor gap, normalized ascending rank, reciprocal match, spatial variation, log(context size)/log(32) |
| Embedding | Linear 84→64, GELU, LayerNorm |
| Attention | Two residual self-attention blocks, 4 heads, width 64, feed-forward 64→128→64, dropout 0 |
| Pooling | One learned seed, 4-head attention, LayerNorm, linear 64→12 |
| Output | Per-anchor/per-branch sigmoid reliability |
| Readout | Two shared scalars: positive softplus slope and bias |
| Trainable count | 90,126 in total |
| Precision | FP32 features, model and statistics |

See [final model](../code/rr_rank_v1/model.py), [feature relations](../src/models/relations.py), and [backbone](../src/models/backbone.py). The network operates on relation descriptors; the original feature vectors are retained for matching.

## Source task sampling

Each update uses one category, four shared query images, and two reference contexts of 16 images. References and queries are disjoint by identity and image hash. The first context is normal. In the second, a sampled number of references is replaced by anomalous images; intact patches in those images remain valid local normal references.

| Anomalous reference count k | 0 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| Probability | 0.10 | 0.20 | 0.30 | 0.25 | 0.10 | 0.05 |

For k>0, random and coherent anomaly sampling are chosen with equal probability. Coherent sampling uses the frozen source anomaly-neighbor inventory. For k=0, the second normal context is disjoint from the first and the queries. Queries follow a shuffled cyclic mixed-label queue with **no fixed normal/anomalous ratio**. Categories are interleaved in shuffled order, with 503 tasks per category per cycle: 12×503 = 6,036 updates. Each cycle covers all source images as queries; some images recur within the fixed-length cycle.

## Training and losses

| Item | Setting |
|---|---|
| Training seed | 17 |
| Reference pretraining | One cycle, 6,036 updates; reliability trained using L_ref; readout fixed |
| Joint training | Four cycles, 24,144 updates; L = L_pix + 0.2 L_ref + 0.1 L_rank |
| Total / selection | 30,180 updates; final scheduled export, no target-score checkpoint selection |
| Optimizer | AdamW, betas (0.9,0.999), eps 1e-8 |
| Learning rate | Each stage: 200-step linear warmup to 2e-4, then cosine decay to 2e-5 |
| Weight decay | 1e-4 on 2-D linear/attention weight matrices; 0 on other parameters |
| Stage transition | Reset optimizer and scheduler at joint stage |
| Gradient clipping | Global norm 1.0 |
| Update unit | One paired task per optimizer update |
| Other | No EMA; deterministic algorithms; TF32 disabled in the shared seeding utility |
| Hardware | Manuscript: one NVIDIA A100-SXM4 40 GB |

`L_ref` uses local normality targets from source mask footprints, balanced between nonempty normal/anomalous anchor groups within each branch and context, then averaged. `L_pix` uses focal loss (positive alpha 0.75, gamma 2) plus 0.5×Dice on anomalous queries only. Its paired-context aggregation is 0.5×mean + 0.5×maximum.

`L_rank` uses margin 0.1 in normalized deviation evidence units. For each normal/anomalous query pair, it compares the largest normal evidence with the smallest anomalous evidence across contexts. The category-specific fixed eta is the fraction of mixed-label tasks in the frozen four-cycle joint plan. Tasks without both labels contribute zero ranking loss. Readout slope/bias are detached in ranking; no image BCE term is active. Numerical guards fail if they would change the declared forward equation rather than silently clipping the ranking score.

The released full optimizer checkpoint is not bundled; the learned inference export is. The archived postflight records completed training and coverage; this is historical evidence rather than a new rerun.

## Matching and target-set inference

For query feature h and reference anchor a with reliability w, the matching cost is `1 - cosine(h,a) + 0.25*(1-w)`. Minimize over anchors **within each reference image**, select the **two distinct images** with the lowest costs, and average over those images and all 12 branches. A shared `sigmoid(softplus(rho)*D + bias)` readout produces the patch probability map, which is bilinearly resized to native resolution with `align_corners=False`.

| Item | Setting |
|---|---|
| Grouping | Within known industrial category or medical domain; label-free balanced sets |
| Nominal size | 20; actual sizes depend on pool balancing |
| Archived constraints | Minimum regular group size 12; rotating implementation accepts group sizes 12–23 |
| Rounds / seeds | Three / 101,202,303 |
| Query rotation | Disjoint query blocks up to four; every image queried once per round |
| References | Other set members outside the active query block; exclude query duplicates and known same-case/instance images |
| Minimum references | At least two after exclusions |
| Model updates | None; no target fitting, checkpoint selection or group normalization |
| Fusion | Average three FP32 native-resolution probability maps pixelwise |
| Image score | Mean of largest ceil(0.01×H×W) values of fused map |
| Per-image min/max scaling | None |

Actual contexts need not contain exactly 16 images after balancing and exclusions. Averaging per-round image scores is a different protocol from scoring the fused map. See [grouping implementation](../code/target14_v1/grouping.py) and [target worker](../code/target518_v1/worker.py).

## Evaluation and scope

Image metrics: AUROC and standard average precision. Localization: pixel AUROC and normalized AUPRO over FPR [0,0.30], called PRO in the manuscript. The frozen protocol specifies all-FP32 pixel scoring, 8-connected regions and 4,096 global PRO thresholds. Dataset summaries use the supplied per-category aggregation; missing/inapplicable values stay NA rather than zero.

The released VisA-source evaluation covers MVTec AD, MPDD, BTAD, DTD-Synthetic, BrainMRI, Br35H, HeadCT, ISIC2016, CVC-ClinicDB, CVC-ColonDB, TN3K and Kvasir-SEG. Real-IAD, KSDD and source-domain VisA self-testing are excluded. The manuscript's separate MVTec AD-source → VisA experiment and ablation checkpoints are not included. See [Results](RESULTS.md) for differences between manuscript and archive.
