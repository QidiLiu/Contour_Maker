# data/

Downloaded/converted datasets live here. **Nothing in this directory is
committed** except this README and the `data/` layout below; everything is
reproducible from the scripts.

```
data/
├── raw/                 ultrasound archives (download cache)
├── interim/             extracted ultrasound datasets
├── unified/             images/ masks/ manifest.csv  (ultrasound, 8362 images)
├── splits/              splits.csv (seed=42, 70/15/15)
├── yolo_det/ yolo_seg/  YOLO exports
├── roi_cache/           Det-UNet MK-UNet ROI cache
├── coco/                RF-DETR COCO export (ultrasound)
└── ade20k/
    ├── raw/             instance-segmentation parquet shards (~1.1 GB)
    ├── unified/         images/ masks/ instances/ manifest.csv
    ├── splits/          splits.csv (4000/600/800)
    ├── yolo_det/ yolo_seg/
    ├── roi_cache/
    └── coco/
```

Build commands: see [`../docs/datasets.md`](../docs/datasets.md) and
[`../docs/reproduce.md`](../docs/reproduce.md).
