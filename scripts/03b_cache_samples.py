"""样本缓存: 把每个目标的多组 (粗糙->精准) 样本预生成并打包存储。

打包方式: 64 张 32x32 ROI 拼成 8x8 的 256x256 灰度图 (uint8), 避免每个 epoch 重复做
Otsu/裁剪/增强, 大幅提升训练吞吐。增强在生成阶段完成 (每个目标生成 variants 组变体)。

    PYTHONPATH=src ./.conda/bin/python scripts/03b_cache_samples.py --level 5 --variants 6 --datasets busi tn3k ddti
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import roi as R  # noqa: E402
from cm.config import DATA, SPLITS, UNIFIED, AugConfig, RefinerConfig  # noqa: E402

CACHE = DATA / "cache"


def quantize_patch(p: np.ndarray) -> np.ndarray:
    """Z-score 后的 patch -> uint8 (固定线性映射, 无损保留 [-4,4] 范围)。"""
    q = np.clip((p + 4.0) / 8.0 * 255.0, 0, 255)
    return np.round(q).astype(np.uint8)


def dequantize_patch(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) / 255.0 * 8.0 - 4.0


def mosaic(patches: np.ndarray, tile: int = 32) -> np.ndarray:
    """(P,S,S) float -> (8*S, 8*S) uint8 拼图。"""
    p = quantize_patch(patches)
    n = p.shape[0]
    side = int(np.ceil(np.sqrt(n)))
    out = np.zeros((side * tile, side * tile), np.uint8)
    for i in range(n):
        r, c = divmod(i, side)
        out[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = p[i]
    return out


def unmosaic(img: np.ndarray, n: int, tile: int = 32) -> np.ndarray:
    side = int(np.ceil(np.sqrt(n)))
    out = np.zeros((n, tile, tile), np.float32)
    for i in range(n):
        r, c = divmod(i, side)
        out[i] = dequantize_patch(img[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--datasets", nargs="*", default=["busi", "tn3k", "ddti"])
    ap.add_argument("--window-ratio", type=float, default=None)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--polarity", default="dark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--out-subdir", default=None, help="缓存子目录名 (默认按参数自动生成)")
    ap.add_argument("--box-init-prob", type=float, default=None,
                    help="用检测框矩形作为初始轮廓的样本比例 (0~1)")
    ap.add_argument("--box-fallback", type=int, default=None, help="Otsu 失败时是否回退到框矩形 (0/1)")
    args = ap.parse_args()

    rcfg = RefinerConfig()
    if args.window_ratio:
        rcfg.window_ratio = args.window_ratio
    if args.n_points:
        rcfg.n_points = args.n_points
    acfg = AugConfig(level=args.level)
    if args.box_init_prob is not None:
        acfg.box_init_prob = float(args.box_init_prob)
    if args.box_fallback is not None:
        acfg.box_fallback = bool(args.box_fallback)
    sub = args.out_subdir or f"L{args.level}_v{args.variants}_wr{int(rcfg.window_ratio*100)}_p{rcfg.n_points}"
    out_dir = CACHE / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["dataset"].isin(args.datasets) & sp["split"].isin(args.splits)]
    print(f"[cache] {len(sp)} 图, level={args.level}, variants={args.variants} -> {out_dir}")

    n_ok = n_fail = 0
    mos_path = out_dir / "mosaics.bin"
    mos_f = open(mos_path, "ab")
    mos_size = mos_path.stat().st_size
    rows: list[dict] = []
    for i, row in enumerate(sp.itertuples(), start=1):
        img = cv2.imread(str(UNIFIED / row.image), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / row.mask), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            n_fail += 1
            continue
        m = (gt > 127).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] < 50:
                continue
            comp = (labels == lab).astype(np.uint8)
            bbox = R.bbox_from_mask(comp)
            if bbox is None:
                continue
            spec = R.TargetSpec(image=img, gt_mask=comp, bbox=bbox, polarity=args.polarity)
            got = 0
            for v in range(args.variants * 3):        # 允许部分失败
                if got >= args.variants:
                    break
                rng = np.random.default_rng(args.seed + i * 1000 + v)
                s = R.build_sample(spec, rcfg, acfg, rng, jitter=True)
                if s is None:
                    continue
                mos = mosaic(s.patches)                       # (256,256) uint8
                mos_f.write(mos.tobytes())
                rows.append(dict(uid=row.uid, dataset=row.dataset, split=row.split,
                                 file=mos_path.name, offset=mos_size, length=mos.nbytes,
                                 shape_h=int(img.shape[0]), shape_w=int(img.shape[1]),
                                 rough=s.rough_norm.astype(np.float16).tobytes(),
                                 target=s.target_norm.astype(np.float16).tobytes(),
                                 aux=s.aux.astype(np.float16).tobytes(),
                                 centers=np.stack([f.center_norm for f in s.frames]).astype(np.float16).tobytes(),
                                 windows=np.array([f.scale for f in s.frames], np.float16).tobytes(),
                                 bbox=np.asarray(s.meta["bbox"], np.float16).tobytes(),
                                 exact=s.meta["exact_global"].astype(np.float16).tobytes(),
                                 roughg=s.rough_global.astype(np.float16).tobytes()))
                mos_size += mos.nbytes
                got += 1
            if got:
                n_ok += 1
        if i % 200 == 0:
            print(f"  {i}/{len(sp)} samples={n_ok} fail={n_fail}", flush=True)
    mos_f.close()

    cache = pd.DataFrame(rows)
    cache.to_parquet(out_dir / "samples.parquet", index=False)
    (out_dir / "meta.json").write_text(
        pd.Series(dict(level=args.level, variants=args.variants,
                       datasets=",".join(args.datasets), splits=",".join(args.splits),
                       window_ratio=rcfg.window_ratio, n_points=rcfg.n_points,
                       roi_size=rcfg.roi_size, polarity=args.polarity,
                       n_targets=n_ok, n_fail=n_fail, n_samples=len(cache))).to_json(indent=2))
    print(f"[cache] 完成 targets={n_ok} samples={len(cache)} fail={n_fail} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
