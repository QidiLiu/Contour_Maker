# Changelog

All notable changes to this project are documented here.
The format roughly follows [Keep a Changelog](https://keepachangelog.com/).

## [3.0.0] - 2026-11-04

### Added

- **RF-DETR-Seg-Nano** as the third scheme: PyTorch training/validation/ONNX
  export (`rfdetr-seg-train/`) and a C++/MNN pipeline (`rfdetr-seg-test/`).
- **ADE20K-10** generalisation benchmark: 10 indoor classes, 4000/600/800
  train/val/test, instance-id masks + class sidecars, YOLO and COCO exports.
- Shared Python evaluation loop (`common/eval.py`) and C++ evaluation harness
  (`cxx-common/eval_harness.cpp`, `summary.json`, `--gt-instances`).
- `results/` collector (`scripts/collect_results.py`).
- Documentation set under `docs/` (datasets, architecture, deployment,
  reproduce, benchmarks), `LICENSE`, `THIRD_PARTY_NOTICES.md`,
  `CONTRIBUTING.md`, `CITATION.cff`.
- `tests/` for metrics, instance decoding and COCO export; GitHub Actions CI
  for lint + tests.

### Changed

- Renamed the two existing schemes to concise, model-based names:
  - one-stage → **YOLO-Seg** (`yolo-seg-train/`, `yolo-seg-test/`, `yolo_seg_test`)
  - two-stage → **Det-UNet** (`det-unet-train/`, `det-unet-test/`, `det_unet_test`)
- Made data export/splitting/validation scripts dataset-agnostic via
  `--dataset`, `--unified`, `--splits-csv`, `--gt-instances`, `--classes`.
- Removed duplicated per-scheme evaluation code (Python and C++).
- `pyproject.toml`: optional `rfdetr`/`dev` extras, package metadata, pytest
  and ruff configuration.

### Fixed

- Det-UNet C++/Python ROI class/score alignment when a box is skipped.
- ADE20K instance classes are preserved end-to-end instead of collapsing to the
  image's primary class.
- `common/metrics.py` no longer duplicates the mask/contour helpers from
  `common/roi.py`.

## [2.0.0] - 2025-10-07

- Two-scheme architecture (one-stage YOLO26n-seg vs two-stage YOLO26n-det +
  MK-UNet), C++/MNN deployment, Python↔MNN cross-validation.

## [1.0.0] - 2025-09

- Initial experiment with YOLO26n + Otsu + contour refiner.
