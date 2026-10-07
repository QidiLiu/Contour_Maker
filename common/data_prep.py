"""数据准备: 下载 → 解压 → 统一命名 → 生成 manifest.csv。

用法:
    python -m common.data_prep --download          # 仅下载 4 个压缩包
    python -m common.data_prep --download --extract # 下载并解压
    python -m common.data_prep --build             # 统一 + 生成 manifest
    python -m common.data_prep --all
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import (
    BUSI_DIRNAME_TO_CLASS,
    BUSI_EXCLUDE_DIRNAMES,
    DATASETS,
    DATASET_BY_KEY,
    INTERIM,
    MASK_THRESHOLD,
    MIN_MASK_AREA,
    RAW,
    UNIFIED,
)

SENTINEL = ".extracted"


# ---------------------------------------------------------------- 下载
def download_one(spec, force: bool = False) -> Path:
    """下载单个压缩包(带 .part 原子落盘 + 断点续传)。"""
    dest = RAW / spec.zip_name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] {spec.zip_name} 已存在 ({dest.stat().st_size/1e6:.1f} MB)")
        return dest

    import requests  # 延迟导入, 仅下载时需要

    part = dest.with_suffix(dest.suffix + ".part")
    url = spec.url
    print(f"  [get ] {spec.key}: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = part.stat().st_size if part.exists() else 0
        mode = "ab" if done else "wb"
        if done and total and done >= total:
            part.rename(dest)
            return dest
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100.0 * done / total
                        print(f"\r        {pct:5.1f}%  {done/1e6:7.1f}/{total/1e6:.1f} MB",
                              end="", flush=True)
    print()
    part.rename(dest)
    print(f"  [ok  ] {spec.zip_name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def download_all(force: bool = False) -> None:
    missing = [s for s in DATASETS if s.key not in {d.key for d in DATASETS}]
    for spec in DATASETS:
        download_one(spec, force=force)


# ---------------------------------------------------------------- 解压
def extract_one(spec, force: bool = False) -> Path:
    zpath = RAW / spec.zip_name
    if not zpath.exists():
        raise FileNotFoundError(f"压缩包不存在, 请先 --download: {zpath}")
    out = INTERIM / spec.key
    sent = out / SENTINEL
    if sent.exists() and not force:
        print(f"  [skip] {spec.key} 已解压")
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"  [unzip] {spec.zip_name} -> {out}")
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            # 阻断 zip-slip
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                print(f"    [warn] 跳过可疑路径 {name}")
                continue
            zf.extract(info, out)
    sent.write_text("ok\n")
    return out


def extract_all(force: bool = False) -> None:
    for spec in DATASETS:
        extract_one(spec, force=force)


# ---------------------------------------------------------------- 目录布局探测
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_HINT = ("_mask", "_segmentation", "_seg", "_label")

# 双扩展名去重: test1.PNG.png 与 test1.png 内容相同，只保留后者。
# 这些是上游数据集打包时的冗余，必须剔除，否则同一张图会被计两次。
_DOUBLE_EXT_RE = re.compile(r"\.([A-Za-z0-9]{2,4})\.(png|jpg|jpeg|bmp|tif|tiff)$", re.I)


def is_mask_file(p: Path) -> bool:
    stem = p.stem.lower()
    return any(h in stem for h in MASK_HINT)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXT


def has_double_ext(p: Path) -> bool:
    """test1.PNG.png -> True"""
    return bool(_DOUBLE_EXT_RE.match(p.name))


def strip_mask_suffix(stem: str) -> str:
    """benign (1)_mask -> benign (1)"""
    low = stem.lower()
    for h in MASK_HINT:
        if low.endswith(h):
            return stem[: -len(h)]
    return stem


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """在解压目录中配对 image / mask。

    两种布局:
      A. 同目录 _mask 后缀 (BUSI):  benign (1).png  +  benign (1)_mask.png
      B. 并行 img/ label/ 目录 (TN3K/DDTI/TG3K): img/x.png + label/x.png

    两条规则同时生效，结果按 (原图, 掩码) 去重。
    """
    out: list[tuple[Path, Path]] = []

    # ---- 布局 A: 同目录 _mask 后缀
    for dirpath, _dirnames, filenames in os_walk(root):
        d = Path(dirpath)
        for fn in filenames:
            p = d / fn
            if not is_image_file(p):
                continue
            base = strip_mask_suffix(p.stem)
            if base == p.stem:
                continue  # 不是 mask
            if has_double_ext(p):
                continue  # xxx_mask_1.png 之类的冗余副本
            img_p = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                q = d / (base + ext)
                if q.exists():
                    img_p = q
                    break
            if img_p is not None:
                out.append((img_p, p))

    # ---- 布局 B: 并行 img/ label/ (或 images/ masks/) 目录
    for dirpath, dirnames, _ in os_walk(root):
        d = Path(dirpath)
        sub = {x.lower(): x for x in dirnames}
        img_dir = next((sub[k] for k in ("img", "images", "image") if k in sub), None)
        msk_dir = next((sub[k] for k in ("label", "labels", "mask", "masks") if k in sub), None)
        if not (img_dir and msk_dir):
            continue
        idir, mdir = d / img_dir, d / msk_dir

        masks: dict[str, Path] = {}
        for p in sorted(mdir.rglob("*")):
            if p.is_file() and is_image_file(p):
                masks[p.stem] = p          # label 文件名与 img 完全同名

        # img 目录内一个逻辑样本可能有多个物理文件
        # (test1.png 与 test1.PNG.png 内容相同)，按去扩展名的 key 只取一个。
        seen_keys: set[str] = set()
        for p in sorted(idir.rglob("*")):
            if not (p.is_file() and is_image_file(p)):
                continue
            # test1.PNG.png -> test1 ; test1.png -> test1
            key = Path(_DOUBLE_EXT_RE.sub(r".\2", p.name)).stem
            if key in masks and key not in seen_keys:
                seen_keys.add(key)
                out.append((p, masks[key]))

    return dedup_pairs(out)


def os_walk(root: Path):
    import os
    return os.walk(root)


def dedup_pairs(pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    seen: set[tuple[str, str]] = set()
    res = []
    for i, m in pairs:
        k = (str(i.resolve()), str(m.resolve()))
        if k in seen:
            continue
        seen.add(k)
        res.append((i, m))
    return res


# ---------------------------------------------------------------- 类别解析
def resolve_class(spec, img_path: Path, gt_mask: np.ndarray) -> tuple[int, str]:
    """返回 (class_id, 说明)。"""
    if not spec.class_from_dirname:
        return spec.class_id, spec.key

    # BUSI: 从路径中找 benign / malignant
    parts = [p.lower() for p in img_path.parts]
    for p in reversed(parts):
        if p in BUSI_DIRNAME_TO_CLASS:
            return BUSI_DIRNAME_TO_CLASS[p], f"dirname:{p}"

    # 回退: 用文件名开头 (benign (1).png)
    stem = img_path.stem.lower()
    for name, cid in BUSI_DIRNAME_TO_CLASS.items():
        if stem.startswith(name):
            return cid, f"stem:{name}"

    # 兜底: 按掩码极性猜不出类别, 记为良性并告警
    return CLS_FALLBACK_BUSI, "fallback:benign"


CLS_FALLBACK_BUSI = BUSI_DIRNAME_TO_CLASS["benign"]


# ---------------------------------------------------------------- 统一化
def normalize_pair(img_p: Path, msk_p: Path):
    """读图并归一化, 返回 (image_gray_u8, mask_u8) 或 None。"""
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        return None
    if img.shape[:2] != msk.shape[:2]:
        msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (msk > MASK_THRESHOLD).astype(np.uint8)
    # 去小连通域
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        keep = np.zeros_like(m)
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] >= MIN_MASK_AREA:
                keep[labels == lab] = 1
        m = keep
    if m.sum() < MIN_MASK_AREA:
        return None
    return img, m


def build_one(spec, force: bool = False) -> pd.DataFrame:
    """解压目录 -> unified images/masks + manifest 行。"""
    root = INTERIM / spec.key
    if not root.exists():
        raise FileNotFoundError(f"未解压: {root}, 请先 --extract")

    out_img = UNIFIED / "images"
    out_msk = UNIFIED / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(root)
    if spec.key == "busi":
        n_before = len(pairs)
        pairs = [(i, m) for i, m in pairs
                 if not any(part.lower() in BUSI_EXCLUDE_DIRNAMES for part in i.parts)]
        print(f"  [pair] {spec.key}: {len(pairs)} 对 "
              f"(排除 normal 无标注类 {n_before - len(pairs)} 张)")

    rows = []
    n_bad = 0
    n_cls_fallback = 0
    for i, m in pairs:
        cid, how = resolve_class(spec, i, None)
        if how.startswith("fallback"):
            n_cls_fallback += 1
        norm = normalize_pair(i, m)
        if norm is None:
            n_bad += 1
            continue
        img, mask = norm
        h, w = img.shape[:2]

        key = f"{spec.key}_{i.stem}"
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        dst_i = out_img / f"{safe}.png"
        dst_m = out_msk / f"{safe}.png"
        cv2.imwrite(str(dst_i), img)
        cv2.imwrite(str(dst_m), (mask * 255).astype(np.uint8))

        rows.append(dict(
            uid=safe, dataset=spec.key, modality=spec.modality, class_id=cid,
            class_source=how, image=f"images/{safe}.png", mask=f"masks/{safe}.png",
            height=h, width=w, src_image=str(i.relative_to(root)), src_mask=str(m.relative_to(root)),
        ))

    df = pd.DataFrame(rows)
    msg = f"  [ok  ] {spec.key}: {len(df)} 张 (丢弃 {n_bad}"
    if n_cls_fallback:
        msg += f", 类别回退 {n_cls_fallback}"
    msg += ")"
    print(msg)
    return df


def build_all(force: bool = False) -> pd.DataFrame:
    frames = []
    for spec in DATASETS:
        frames.append(build_one(spec, force=force))
    df = pd.concat(frames, ignore_index=True)

    # uid 全局去重
    before = len(df)
    df = df.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)
    if len(df) != before:
        print(f"  [dedup] uid 去重: {before} -> {len(df)}")

    UNIFIED.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIFIED / "manifest.csv", index=False)

    print("\n=== 类别分布 ===")
    tab = pd.crosstab(df["dataset"], df["class_id"])
    print(tab)
    print(f"\n总计 {len(df)} 张 -> {UNIFIED / 'manifest.csv'}")
    return df


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="数据准备")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    if not any([args.download, args.extract, args.build, args.all]):
        ap.error("至少指定 --download / --extract / --build / --all 之一")

    if args.download or args.all:
        print("=== 下载 ===")
        download_all(force=args.force)
    if args.extract or args.all:
        print("=== 解压 ===")
        extract_all(force=args.force)
    if args.build or args.all:
        print("=== 统一 + manifest ===")
        build_all(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())