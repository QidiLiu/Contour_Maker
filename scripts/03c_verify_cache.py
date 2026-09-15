"""缓存完整性校验: 检查 mosaics.bin 与 samples.parquet 是否一致、能否正确重建 ROI 与轮廓。

    PYTHONPATH=src ./.conda/bin/python scripts/03c_verify_cache.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import DATA  # noqa: E402


def _f16(b) -> np.ndarray:
    return np.frombuffer(b, np.float16).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=str(DATA / "cache"))
    args = ap.parse_args()
    root = Path(args.cache_root)
    ok_all = True
    for d in sorted(root.glob("L*_v*")):
        pq, binp, meta = d / "samples.parquet", d / "mosaics.bin", d / "meta.json"
        if not pq.exists():
            print(f"[skip] {d.name}: 缺少 samples.parquet")
            continue
        df = pd.read_parquet(pq)
        size = binp.stat().st_size
        raw = binp.read_bytes()
        bad_off = int((df["offset"] + df["length"] > size).sum())
        # 抽样重建校验
        rng = np.random.default_rng(0)
        idx = rng.choice(len(df), size=min(20, len(df)), replace=False)
        bbox_ok = 0
        pt_ok = 0
        for i in idx:
            r = df.iloc[i]
            img = np.frombuffer(raw, np.uint8, int(r["length"]), int(r["offset"]))
            side = 8
            patch = img.reshape(side * 32, side * 32)[:32, :32].astype(np.float32) / 255 * 8 - 4
            if 0.05 < patch.std() < 6.0:
                pt_ok += 1
            bb = _f16(r["bbox"])
            if bb[2] > bb[0] and bb[3] > bb[1] and bb[0] >= 0 and bb[1] >= 0:
                bbox_ok += 1
        tgt = _f16(df.iloc[0]["target"]).reshape(-1, 2)
        rough = _f16(df.iloc[0]["rough"]).reshape(-1, 2)
        inrange = float(np.mean(np.abs(tgt) <= 1.0))
        status = "OK" if bad_off == 0 and pt_ok == len(idx) and bbox_ok == len(idx) else "FAIL"
        ok_all &= status == "OK"
        print(f"[{status}] {d.name}: samples={len(df)} bin={size/1e6:.0f}MB 越界={bad_off} "
              f"patch_ok={pt_ok}/{len(idx)} bbox_ok={bbox_ok}/{len(idx)} "
              f"首样本 target 在 [-1,1] 内比例={inrange:.3f} rough_pts={len(rough)}")
    print("全部通过" if ok_all else "存在问题")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
