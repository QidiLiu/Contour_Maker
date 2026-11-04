# Third-party notices

This project builds on the following open-source projects. Their licenses apply
to their respective code, weights and data.

| Project | Version / commit | License | Used for |
|---|---|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | 8.4.174 | AGPL-3.0 | YOLO26n / YOLO26n-seg training, export, inference |
| [RF-DETR](https://github.com/roboflow/rf-detr) | 1.11.2 | Apache-2.0 | RF-DETR-Seg-Nano training, export, inference |
| [MNN](https://github.com/alibaba/MNN) | 3.4.0 | Apache-2.0 | C++ inference runtime |
| [MK-UNet](https://github.com/SLDGroup/MK-UNet) | ICCV 2025 CVAMD | MIT | Network architecture (re-implemented in `common/mkunet.py`) |
| [ADE20K](https://sceneparsing.csail.mit.edu/) | 2017 | BSD-3-Clause | Generalisation benchmark, downloaded at build time |
| BUSI | — | CC BY 4.0 (upstream) | Ultrasound breast lesions |
| TN3K / DDTI / TG3K | — | see upstream | Ultrasound thyroid data |

## Notes

- The project itself is distributed under **AGPL-3.0** because the YOLO training
  and inference path uses Ultralytics.
- `common/mkunet.py` is an independent re-implementation of the MK-UNet
  architecture: it removes the `timm` dependency, supports `in_channels=1` and
  fixes the attention double-application bug described in the upstream code.
  No upstream training scripts or weights are copied.
- `cxx-common/src/png.cpp` is a minimal self-contained PNG codec written for this
  project (no third-party code).
- Dataset files are **not** redistributed. The scripts download from the original
  mirrors and convert them locally.
- RF-DETR pretrained weights are downloaded by the `rfdetr` package and remain
  subject to Apache-2.0.
