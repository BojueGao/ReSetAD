# Setup and reproduction

## What works in a fresh checkout

Read the code, inspect the archived CSV results and task plan, and verify source integrity without a GPU or third-party Python packages:

```bash
python tools/verify_source_release.py
```

The original manifest lists 79 scientific files. Newly added documentation is inventoried separately. Source verification is not a scientific replication.

## Original environment

The snapshot records these observed versions in [requirements-observed.txt](../requirements-observed.txt):

| Package | Recorded version |
|---|---|
| PyTorch | 2.9.0+cu128 |
| torchvision | 0.24.0+cu128 |
| NumPy | 2.5.2 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| PyYAML | 6.0.3 |
| timm | 1.0.22 |
| pytest | 8.3.3 |

These are archive observations, not a validated install recipe or an availability guarantee. The supplied requirements file does not record the Python version or the complete dependency graph. In particular, a default package index may not resolve the CUDA-tagged PyTorch build. Establish a compatible Linux/CUDA environment using the package vendors' official installation instructions and record any version differences as a new reproduction environment. Do not present a successful package installation as a reproduction of the experiment.

## External assets

1. Obtain the official non-register DINOv2 ViT-L/14 implementation and pretrained weights under their terms. The supplied backbone adapter imports `dinov2.hub.backbones` from a local checkout.
2. Obtain source VisA images, its original metadata (`split_csv/1cls.csv`) and anomaly masks. Final source training merges the original partitions and uses all 12 categories.
3. Obtain the desired target datasets and their native-resolution masks where available. Preserve known case IDs and decoded-image hashes for grouping exclusions.
4. Restore or independently rebuild the required audited manifests and FP32 feature caches, recording their provenance. For an exact archived replay, the original file hashes, neighbor inventory, task identities and execution records must match.

Official backbone URL recorded in the archive:

<https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth>

```text
Expected backbone SHA256
d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428
```

See [backbone provenance](../provenance/backbone_weight.json) for the recorded inventory. The backbone file is not the small learned inference export bundled with this repository.

## Archived workspace dependencies

The original Linux workspace was `/home/ubuntu/MyDisk/VSI-AD-Large-518`, using `/home/ubuntu/MyDisk/VSI-AD/.venv/bin/python`. This is provenance, not a portable installation path.

The following assets are referenced but are not a complete part of this release:

| Dependency | Examples |
|---|---|
| Source records/cache | `outputs/manifests/source_full.jsonl`, `outputs/cache/source_518/` |
| Target audit and inputs | `result/dataset_audit/<dataset>/`, target input manifests |
| Target feature cache and predictions | External cache files and native prediction maps |
| Historical contracts | Files named in `provenance/frozen_inputs.json` and original workspace code |
| Lifecycle/budget records | Engineering acceptance, parent contracts and cumulative ledger |
| Full training checkpoint | `result/rr_rank_v1/checkpoints/step_030180.pt` (external) |

The frozen task plan, neighbor inventory, source postflight, final learned head and summary metrics are included. Some receipt hashes refer to evidence retained only on the original server; copying their hashes does not provide those missing files.

Do not launch archived supervisors as a fresh-checkout quick start: their completeness gates will fail without these inputs, and their original output destinations already describe a completed experiment. A new reproduction needs a separate output directory and newly established execution contracts. Preserve the original snapshot when adapting paths or interfaces.

## Using the learned head in an integration

The final implementation is `code/rr_rank_v1/model.py`, class `RR`, rather than the historical `src/models/model.py`. After installing compatible dependencies, an integration can put the repository root and `code/rr_rank_v1` on Python's module path, construct `RR()`, and load the `model` state dict from the supplied export. Use only the supplied trusted checkpoint and verify its SHA256 first. This loading path was not executed during documentation preparation.

The frozen backbone emits features shaped `[N,4,37,37,1024]`. `RR.prepare_context(context_features)` builds a temporary reference state; `RR.score_queries(state, query_features, sizes)` returns native maps and scores. `sizes` must contain native `(height,width)` tuples. Contexts need at least two eligible reference images. Run inference in evaluation mode without gradients. This API operates on frozen features; it is not an image-folder command-line program.

`code/rr_rank_v1/inference.py` provides label-rejecting grouping/prediction helpers. Dataset-level evaluation must implement all three seeded partitions, query rotation, duplicate/case exclusions and native-map fusion described in [Experiment settings](EXPERIMENT_SETTINGS.md). A single arbitrary context forward call does not reproduce the dataset protocol.

## Reproducing the full experiment

Follow this order in an independently prepared workspace:

1. Audit source and target data and establish manifests with counts, native geometry and hashes.
2. Verify the official frozen backbone and extract/cache FP32 features with the exact 518 preprocessing and layer indices.
3. Check source-neighbor ordering, seed-17 paired task identities and category eta against the frozen plan.
4. Validate numerical behavior and exact checkpoint recovery in the new environment.
5. Run 6,036 reference-only updates, reset optimizer/scheduler, and run 24,144 joint updates.
6. Use the final scheduled head at update 30,180. Preserve model, optimizer, scheduler, sampler and RNG state for resumability.
7. Run the fixed source diagnostics, then the 12 target protocols with no target parameter updates.
8. Fuse three native maps per image before top-1% scoring and compute metrics with the frozen definitions.

Steps 1–8 describe the required protocol, not a claim that this archive supplies a turnkey launcher or all execution assets. The separate reverse-source model needed for the manuscript's VisA row and the manuscript ablation runs are absent.

## Troubleshooting

| Symptom | Meaning / next action |
|---|---|
| Missing manifest, ledger or parent-contract error | An archived execution dependency is absent; restore the evidence or establish a separately documented reproduction contract |
| Register-weight or feature-shape error | Use non-register DINOv2-L/14 with the four fixed layers and 518×518 preprocessing |
| Dataset metrics are NA | The corresponding image labels or real pixel masks are inapplicable; do not fill with zero |
| Paper score differs from CSV | See [Results](RESULTS.md); do not overwrite archive metrics |
| Source hash verification fails | Restore the original scientific bytes; path/config adaptations belong to a separately recorded reproduction |
