"""准备超声数据集: 解压 raw zip -> data/interim -> 配对 image/mask -> data/unified。

用法:
    PYTHONPATH=src ./.conda/bin/python scripts/01_prepare_data.py [--only busi]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm.config import DATASETS, INTERIM, RAW, UNIFIED, DatasetSpec  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


# --------------------------------------------------------------- unzip
def extract(spec: DatasetSpec) -> Path:
    """解压到 data/interim/<key>, 返回解压根目录。"""
    dst = INTERIM / spec.key
    done_flag = dst / ".extracted"
    if done_flag.exists():
        log(f"{spec.key}: 已解压, 跳过")
        return dst

    src = RAW / spec.zip_name
    if not src.exists():
        raise FileNotFoundError(f"缺少 {src}, 请先下载数据集")
    dst.mkdir(parents=True, exist_ok=True)
    log(f"{spec.key}: 解压 {src.name} ({src.stat().st_size / 1e6:.1f} MB)")
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dst)
    done_flag.write_text("ok")
    return dst


# --------------------------------------------------------------- pairing
def _is_mask_name(p: Path) -> bool:
    n = p.stem.lower()
    return n.endswith("_mask") or n.endswith("-mask") or n.endswith("_seg") or n.endswith("_segmentation")


def _norm_key(p: Path) -> str:
    """把 mask 路径映射成与之配对的图像 stem key (小写)。"""
    n = p.stem.lower()
    for suf in ("_mask", "-mask", "_seg", "_segmentation"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    # DDTI 等数据集存在 test1.PNG.png 这类双扩展名
    for suf in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def _pair_img_label_dirs(root: Path) -> list[tuple[Path, Path]]:
    """布局: <root>/img/*.png 与 <root>/label/*.png 按 stem 配对。"""
    pairs: list[tuple[Path, Path]] = []
    for img_dir_name, lab_dir_name in (("img", "label"), ("images", "labels"),
                                       ("image", "label"), ("img", "labels"), ("images", "label")):
        d_img, d_lab = root / img_dir_name, root / lab_dir_name
        if not (d_img.is_dir() and d_lab.is_dir()):
            continue
        labs = {_norm_key(p): p for p in d_lab.iterdir() if p.suffix.lower() in IMG_EXT}
        imgs: dict[str, Path] = {}
        for p in d_img.iterdir():
            if p.suffix.lower() not in IMG_EXT or _is_mask_name(p):
                continue
            k = _norm_key(p)
            # 双扩展名时优先选"更规范"的那个
            if k not in imgs or len(p.name) < len(imgs[k].name):
                imgs[k] = p
        for k, m in labs.items():
            if k in imgs:
                pairs.append((imgs[k], m))
    return pairs


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """在解压目录中寻找 (image, mask) 配对。支持常见 5 种布局。

    返回按优先级去重后的配对: 同一 mask 只保留最可靠的图像来源。
    """
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and not _is_mask_name(p)]
    masks = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and _is_mask_name(p)]
    # (优先级, image, mask)  优先级越小越可靠
    tagged: list[tuple[int, Path, Path]] = []

    # 布局 A: 任意层级的 <dir>/img/*.png 与 <dir>/label/*.png 配对
    dirs_to_try: set[Path] = {root}
    for p in imgs + masks:
        d = p.parent
        while True:
            dirs_to_try.add(d)
            if d == root or d.parent == d:
                break
            d = d.parent
    for d in dirs_to_try:
        if d.name.lower() in ("img", "image", "images", "label", "labels", "mask", "masks"):
            continue
        for a, b in _pair_img_label_dirs(d):
            tagged.append((1, a, b))

    if masks:
        # 布局 B: 同目录 _mask 后缀
        for m in masks:
            cand = m.with_name(_norm_key(m) + m.suffix)
            if m.suffix.lower() == cand.suffix.lower() and cand.exists() and cand != m:
                tagged.append((0, cand, m))
                continue
            # 布局 C: 平行目录 images/ <-> masks/
            parts = list(m.parts)
            hit = False
            for i, part in enumerate(parts):
                low = part.lower()
                if low in ("mask", "masks", "mask_gt", "ground_truth", "gt", "label", "labels",
                           "img", "image", "images"):
                    repl = {"mask": "image", "masks": "images", "mask_gt": "image",
                            "ground_truth": "image", "gt": "image", "label": "image",
                            "labels": "images", "img": "img", "image": "img",
                            "images": "img"}[low]
                    for alt in (repl, repl + "s", "img", "IMG", "image", "images", "Image", "Images"):
                        trial = Path(*parts[:i], alt, *parts[i + 1:])
                        if trial.exists():
                            tagged.append((0, trial, m))
                            hit = True
                            break
                    if hit:
                        break
            if hit:
                continue
            # 兜底: 按规范化 stem 匹配 (双扩展名等)
            cands = [p for p in imgs if _norm_key(p) == _norm_key(m)]
            if len(cands) == 1:
                tagged.append((1, cands[0], m))

    # 布局 D: 目录名区分 (benign/ <-> benign_mask/)
    for d in sorted({p.parent for p in imgs}):
        for suffix in ("_mask", "_masks", "-mask", "_gt", "_label"):
            cand_dir = d.with_name(d.name + suffix)
            if cand_dir.is_dir():
                for img in sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT):
                    for ext in (img.suffix, ".png", ".jpg", ".tif", ".bmp"):
                        m = cand_dir / (img.stem + ext)
                        if m.exists():
                            tagged.append((1, img, m))
                            break
                break

    # 去重: 同一 mask 只保留最高优先级; 同一 image 只保留一次
    tagged.sort(key=lambda t: t[0])
    seen_m: set[str] = set()
    seen_i: set[str] = set()
    out: list[tuple[Path, Path]] = []
    for _pri, a, b in tagged:
        ka, kb = str(a.resolve()), str(b.resolve())
        if ka in seen_i or kb in seen_m:
            continue
        seen_i.add(ka); seen_m.add(kb)
        out.append((a, b))
    return sorted(out)


# --------------------------------------------------------------- normalise
def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只处理指定数据集 key")
    args = ap.parse_args()

    out_img = UNIFIED / "images"
    out_msk = UNIFIED / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in DATASETS:
        if args.only and spec.key not in args.only:
            continue
        root = extract(spec)
        pairs = find_pairs(root)
        log(f"{spec.key}: 发现 {len(pairs)} 对 image/mask")
        kept = 0
        for img_p, msk_p in pairs:
            img = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
            msk = cv2.imread(str(msk_p), cv2.IMREAD_UNCHANGED)
            if img is None or msk is None:
                continue
            img = to_gray_u8(img)
            msk = to_gray_u8(msk)
            if msk.shape != img.shape:
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            msk = (msk > 127).astype(np.uint8)
            if msk.sum() < 50:      # 忽略空 mask (无目标/标注缺失)
                continue
            uid = f"{spec.key}_{hashlib.md5(str(img_p).encode()).hexdigest()[:10]}"
            cv2.imwrite(str(out_img / f"{uid}.png"), img)
            cv2.imwrite(str(out_msk / f"{uid}.png"), msk * 255)
            rows.append(
                dict(uid=uid, dataset=spec.key, modality=spec.modality,
                     image=f"images/{uid}.png", mask=f"masks/{uid}.png",
                     height=int(img.shape[0]), width=int(img.shape[1]),
                     src_image=str(img_p.relative_to(INTERIM)))
            )
            kept += 1
        log(f"{spec.key}: 保留 {kept} 张 (含非空 mask)")

    df = pd.DataFrame(rows)
    manifest = UNIFIED / "manifest.csv"
    df.to_csv(manifest, index=False)
    stats = df.groupby(["dataset", "modality"]).size().to_dict()
    (UNIFIED / "stats.json").write_text(json.dumps(
        {"total": int(len(df)), "by_dataset": {f"{k[0]}": int(v) for k, v in stats.items()},
         "image_sizes": df.groupby("dataset")[["height", "width"]].median().to_dict()},
        indent=2, ensure_ascii=False))
    log(f"完成: {len(df)} 张 -> {manifest}")
    print(df.groupby(["dataset", "modality"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
