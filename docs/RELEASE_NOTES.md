# GitHub preparation notes

Prepared on 2026-11-18 from the supplied `ReSetAD_L518_github_ready.zip` and manuscript PDF.

## Preserved material

The 79 files in `provenance/source_release_manifest.json` remain byte-identical to the supplied archive. This includes scientific code, frozen configs, learned head, task plan, source-neighbor inventory, diagnostics, source training records and metric CSV files. No model was retrained, no metric was recomputed, and no new software license was assigned.

The supplied README is preserved as [SNAPSHOT_README.md](SNAPSHOT_README.md). The manuscript is copied without modification as [paper.pdf](paper.pdf). Historical server paths and contract references are retained as provenance; this publication pass does not fill in missing runtime assets.

## Added publication material

- English and Chinese repository introductions.
- Detailed experiment settings traced to manuscript sections and actual code.
- Environment, data, checkpoint and reproduction requirements.
- Archived metric tables and an explicit comparison with manuscript Table 1.
- `configs/paper_settings.json`, a machine-readable documentation summary that does not change archived launchers.
- A CPU-only GitHub Actions source-integrity check.
- Root-anchored dataset/weight ignore rules so `src/data/` remains included in Git.
- `provenance/publication_manifest.json`, recording checksums for the prepared release (excluding itself).

## Remaining limitations

This is a single-seed VisA-source snapshot, not the complete 13-benchmark manuscript release. The separate MVTec AD-source model, ablation checkpoints, original datasets, full feature caches, native prediction maps, complete server receipts and full optimizer checkpoint are not bundled. The manuscript and archive contain different values for some metrics. No cause for these differences is established by the provided files.

The package versions were observed in the source environment; a new installation and GPU execution were not validated during this preparation. Source hashing and Python parsing validate archive integrity, not scientific correctness or full reproducibility.
