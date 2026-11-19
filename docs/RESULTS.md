# Archived results and manuscript comparison

## Archived VisA-source snapshot

Values below are percentages, copied from the supplied per-dataset CSV and rounded to two decimals for display. They are **not new evaluations**. Original precision is retained in [metrics_per_dataset.csv](../result/inference/large518_target12_v1/metrics/metrics_per_dataset.csv); category-level values are in [metrics_per_category.csv](../result/inference/large518_target12_v1/metrics/metrics_per_category.csv).

| Dataset | Images | Categories | Image AUROC | Image AP | Pixel AUROC | AUPRO@0.30 |
|---|---:|---:|---:|---:|---:|---:|
| MVTec AD | 1725 | 15 | 94.08 | 97.35 | 97.99 | 95.62 |
| MPDD | 458 | 6 | 71.17 | 73.19 | 97.33 | 93.37 |
| BTAD | 741 | 3 | 94.55 | 96.96 | 98.33 | 83.18 |
| DTD-Synthetic | 1304 | 12 | 98.01 | 99.22 | 99.13 | 97.43 |
| BrainMRI | 253 | 1 | 89.10 | 91.05 | NA | NA |
| Br35H | 3000 | 1 | 90.09 | 87.61 | NA | NA |
| HeadCT | 200 | 1 | 83.80 | 74.44 | NA | NA |
| ISIC2016 | 379 | 1 | NA | NA | 73.58 | 29.60 |
| CVC-ClinicDB | 612 | 1 | NA | NA | 85.10 | 56.34 |
| CVC-ColonDB | 380 | 1 | NA | NA | 75.07 | 44.78 |
| TN3K | 614 | 1 | NA | NA | 79.43 | 63.45 |
| Kvasir-SEG | 1000 | 1 | NA | NA | 74.06 | 29.22 |

Image-only medical datasets lack real pixel masks. Single-label localization pools do not support image AUROC/AP. NA means inapplicable, not zero. These are the CSV's per-dataset summaries; this documentation does not reaggregate category scores. The archive covers 10,666 target images and records 31,998 round maps in the historical delivery summary; native map files are not bundled.

## Difference from manuscript Table 1

The [manuscript](paper.pdf), page 6, reports 13 benchmarks and includes VisA results from a separate MVTec AD-source model. That reverse-source model and its metrics are absent from this archive. Several manuscript numbers also differ from the supplied CSV. The table below lists **all displayed differences at two decimal places** for the 12 overlapping datasets.

| Dataset | Metric | Manuscript Table 1 | Archived CSV |
|---|---|---:|---:|
| MVTec AD | Image AUROC | 98.18 | 94.08 |
| MVTec AD | Image AP | 99.15 | 97.35 |
| BTAD | Image AUROC | 96.38 | 94.55 |
| BTAD | Image AP | 98.16 | 96.96 |
| DTD-Synthetic | Image AUROC | 98.64 | 98.01 |
| BrainMRI | Image AUROC | 95.70 | 89.10 |
| BrainMRI | Image AP | 96.85 | 91.05 |
| Br35H | Image AUROC | 96.89 | 90.09 |
| Br35H | Image AP | 95.41 | 87.61 |
| HeadCT | Image AUROC | 96.13 | 83.80 |
| HeadCT | Image AP | 95.84 | 74.44 |
| ISIC2016 | Pixel AUROC | 96.17 | 73.58 |
| ISIC2016 | AUPRO@0.30 | 85.23 | 29.60 |
| CVC-ClinicDB | Pixel AUROC | 95.10 | 85.10 |
| CVC-ClinicDB | AUPRO@0.30 | 76.34 | 56.34 |
| CVC-ColonDB | Pixel AUROC | 85.07 | 75.07 |
| CVC-ColonDB | AUPRO@0.30 | 76.78 | 44.78 |
| Kvasir-SEG | Pixel AUROC | 89.06 | 74.06 |
| Kvasir-SEG | AUPRO@0.30 | 68.22 | 29.22 |

At the displayed precision, all four MPDD metrics, BTAD localization, DTD-Synthetic localization, MVTec AD localization and both TN3K localization metrics agree. The provided material does not establish why the other values differ. No scores have been substituted, reconciled or fabricated. The manuscript's “22 of 36” ranking claim cannot be established from this snapshot alone.

## Interpretation limits

- One training seed (17); target partition seeds 101/202/303 are grouping rounds, not independent training runs.
- Inference accesses other unlabeled images from the target collection and depends on category/domain grouping and reference coverage.
- Earlier target diagnostics informed method development, as disclosed in the supplied snapshot README; this release does not establish a target-blind development process.
- Historical acceptance/postflight records are included, but not every referenced receipt or prediction file.
- Repository preparation verified source bytes and syntax. It did not rerun GPU training, target inference, metric evaluation or manuscript ablations.
