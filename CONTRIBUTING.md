# Contributing

Thanks for considering a contribution! This project compares three instance
segmentation schemes on a shared data/metric protocol, so reproducibility and
fair comparisons matter more than micro-optimisations.

## Development setup

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install torch torchvision
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
VIRTUAL_ENV=$PWD/.venv uv pip install "rfdetr[train,onnx]"
```

## Checks

```bash
.venv/bin/ruff check common yolo-seg-train det-unet-train rfdetr-seg-train scripts tests
.venv/bin/ruff format --check common yolo-seg-train det-unet-train rfdetr-seg-train scripts tests
.venv/bin/pytest -q
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

The C++ build needs `third_party/MNN` (see `docs/deployment.md`); CI only runs
the Python lint/tests because MNN is a large optional submodule.

## Guidelines

- **Python**: Python 3.11, 100-column lines, `ruff` (E/F/I/UP/B). Public helpers
  get type hints and a short docstring. Prefer changing `common/` over copying
  logic into a scheme directory.
- **C++**: C++17, 100 columns, `snake_case` files, `PascalCase` types,
  `kCamelCase` enums. New inference pipelines should implement an `Infer`
  callback and call `cxxcommon::RunEvaluation` instead of re-writing the eval
  loop.
- **Data**: never commit `data/`, `runs/`, `logs/` or `reports/`. The committed
  `models/*.mnn` are the only binary artifacts (plus the small YOLO base
  weights in `weights/`).
- **Comparisons**: any new scheme must be evaluated with `common.eval` (Python)
  and `cxx-common/eval_harness` (C++) on the same split, with the same IoU
  matching threshold and boundary tolerance. Report the input resolution and
  hardware next to the numbers.
- **Docs**: update `README.md` and the relevant `docs/*.md` in the same PR.
  Keep Chinese and English terms consistent with the scheme names
  (YOLO-Seg / Det-UNet / RF-DETR-Seg).

## Adding a scheme

1. Add `<scheme>-train/` (training, validation, ONNX export) and, if it can be
   deployed, `<scheme>-test/` (C++/MNN pipeline).
2. Reuse `common.eval` and `cxxcommon::RunEvaluation`.
3. Add a row to the scheme table in `README.md` and `docs/architecture.md`.
4. Add the binary to the root `CMakeLists.txt`.
5. Run the two datasets (`ultrasound`, `ade20k`) and commit the summary CSV
   under `results/`.

## Commit messages

Short imperative subject, Chinese or English is fine, e.g.:

```
add RF-DETR-Seg-Nano scheme and ADE20K benchmark
fix ROI class alignment in det-unet
```

## Reporting issues

Please include: scheme name, dataset, command, Python/C++ versions, GPU/CPU,
and the `summary.json` (or the printed metrics table). Never paste private data.
