"""在相同训练集上微调 EdgeSAM / EfficientSAM 的 mask decoder (box prompt, 编码器冻结)。

流程:
  1) 用训练好的 YOLO26n 在 train/val 上生成检测框 (与推理一致)
  2) 冻结编码器, 预计算并缓存每个「目标」的图像 embedding
  3) 仅训练 mask decoder: 输入 (embedding + box prompt), 监督 = GT mask
     loss = BCE(Dice) 组合
  4) 保存 decoder 权重 -> weights/sam_finetuned/<variant>_decoder.pt

    PYTHONPATH=src ./.conda/bin/python scripts/19_finetune_sam.py --variant edgesam --epochs 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import metrics as M  # noqa: E402
from cm.config import ROOT, SPLITS, UNIFIED  # noqa: E402
from cm.sam_models import SAM_SIZE, WEIGHTS, _scale_box, _to_rgb_tensor, build_sam  # noqa: E402

OUT_DIR = WEIGHTS / "sam_finetuned"
CACHE_DIR = ROOT / "data" / "cache" / "sam_embed"


def pick_box(boxes, gt_box):
    """选与 GT 框 IoU 最大的检测框; 不达标返回 None。"""
    best, bi = None, 0.0
    for b in boxes:
        i = M.bbox_iou(tuple(map(int, b)), gt_box)
        if i > bi:
            best, bi = b, i
    return (best, bi) if bi >= 0.3 else (None, bi)


def build_cache(variant: str, splits: list[str], det_path: str, conf: float, imgsz: int,
                limit: int = 0) -> Path:
    """生成 (embedding, box, gt_mask) 缓存。embedding 存为 npy (float16)。"""
    from ultralytics import YOLO
    cache_dir = CACHE_DIR / variant
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_f = cache_dir / "meta.csv"
    if meta_f.exists():
        print(f"[cache] 复用已存在的 {meta_f}")
        return cache_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = build_sam(variant, device=str(device))
    det = YOLO(det_path)
    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["split"].isin(splits)]
    if limit:
        sp = sp.head(limit)

    emb_path = cache_dir / "embeddings.npy"
    rows, embs = [], []
    t0 = time.time()
    for n, row in enumerate(sp.itertuples(), 1):
        img = cv2.imread(str(UNIFIED / row.image), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / row.mask), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        H, W = img.shape[:2]
        m = (gt > 127).astype(np.uint8)
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        comps = [((labels == i).astype(np.uint8), M.mask_bbox((labels == i).astype(np.uint8)))
                 for i in range(1, ncomp) if stats[i, cv2.CC_STAT_AREA] >= 50]
        comps = [(c, b) for c, b in comps if b]
        if not comps:
            continue
        res = det.predict(str(UNIFIED / row.image), conf=conf, imgsz=imgsz,
                          verbose=False, device=str(device))[0]
        boxes = ([tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
                 if res.boxes is not None and len(res.boxes) else [])
        if not boxes:
            continue
        with torch.no_grad():
            emb = wrapper.encode(img)
        emb16 = emb[0].to(torch.float16).cpu().numpy()
        for comp, gb in comps:
            b, bi = pick_box(boxes, gb)
            if b is None:
                continue
            rows.append(dict(uid=row.uid, split=row.split, h=H, w=W, box=",".join(map(str, b)),
                             box_iou=bi, idx=len(embs)))
            embs.append(emb16)
        if n % 200 == 0:
            print(f"  {n}/{len(sp)}  样本={len(rows)}  {time.time()-t0:.0f}s", flush=True)

    np.save(emb_path, np.stack(embs))
    pd.DataFrame(rows).to_csv(meta_f, index=False)
    print(f"[cache] {variant}: {len(rows)} 个目标, embedding={emb_path} "
          f"({emb_path.stat().st_size/1e6:.0f}MB, {time.time()-t0:.0f}s)")
    return cache_dir


class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: Path, split: str):
        self.dir = cache_dir
        self.meta = pd.read_csv(cache_dir / "meta.csv")
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.emb = np.load(cache_dir / "embeddings.npy", mmap_mode="r")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        e = torch.from_numpy(np.asarray(self.emb[int(r["idx"])], dtype=np.float32))
        box = tuple(float(v) for v in str(r["box"]).split(","))
        H, W = int(r["h"]), int(r["w"])
        gt = cv2.imread(str(UNIFIED / "masks" / f"{r['uid']}.png"), cv2.IMREAD_GRAYSCALE)
        gm = (gt > 127).astype(np.float32)
        gm256 = cv2.resize(gm, (256, 256), interpolation=cv2.INTER_NEAREST)
        return e, torch.tensor(_scale_box(box, H, W)), torch.from_numpy(gm256)[None], (H, W)


def decoder_forward(wrapper, emb, box_scaled):
    """统一 decoder 调用: 返回低分辨率 logits (B,1,256,256)。

    逐图调用 (EdgeSAM/EfficientSAM 的官方推理路径都是 batch=1), 避免批量语义歧义。
    EdgeSAM 的 image_pe 使用其 image_encoder 输出的 pos_embed (形状与 embedding 相同)。
    """
    outs = []
    for i in range(emb.shape[0]):
        e = emb[i:i + 1]
        b = box_scaled[i:i + 1]
        if wrapper.__class__.__name__ == "EdgeSAMWrapper":
            boxes = b[:, None, :]
            sparse, dense = wrapper.model.prompt_encoder(points=None, boxes=boxes, masks=None)
            low_res, _ = wrapper.model.mask_decoder(
                image_embeddings=e,
                image_pe=wrapper.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                num_multimask_outputs=1)
        else:
            corners = torch.stack([b[:, [0, 1]], b[:, [2, 3]]], dim=1)[:, None, :, :]
            labels = torch.ones(1, 1, 2, dtype=torch.int, device=e.device)
            low_res, _ = wrapper.model.predict_masks(
                image_embeddings=e, batched_points=corners, batched_point_labels=labels,
                multimask_output=False, input_h=SAM_SIZE, input_w=SAM_SIZE,
                output_h=256, output_w=256)
        outs.append(low_res.reshape(1, 1, 256, 256))
    return torch.cat(outs, dim=0)


def dice_loss(logits, target, eps: float = 1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["edgesam", "efficientvit-t", "efficientvit-s"])
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache-limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = build_cache(args.variant, ["train", "val"], args.det, args.conf,
                            args.imgsz, args.cache_limit)
    ds_tr = EmbedDataset(cache_dir, "train")
    ds_va = EmbedDataset(cache_dir, "val")
    print(f"[data] train={len(ds_tr)} val={len(ds_va)}")

    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                                        num_workers=2, drop_last=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                                        num_workers=2)

    wrapper = build_sam(args.variant, device=str(device))
    for p in wrapper.model.parameters():
        p.requires_grad = False
    for p in wrapper.model.mask_decoder.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
    print(f"[model] {wrapper.name}: 可训练参数 {n_train/1e6:.2f}M (仅 mask decoder)")

    opt = torch.optim.AdamW([p for p in wrapper.model.mask_decoder.parameters()], lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        wrapper.model.mask_decoder.train()
        tot, nb = 0.0, 0
        t0 = time.time()
        for emb, box, gm, _hw in dl_tr:
            emb = emb.to(device); box = box.to(device); gm = gm.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = decoder_forward(wrapper, emb, box)
                bce = F.binary_cross_entropy_with_logits(logits, gm)
                loss = bce + dice_loss(logits, gm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()); nb += 1
        sched.step()

        # 验证 (256 分辨率下的 Dice)
        wrapper.model.mask_decoder.eval()
        dices = []
        with torch.no_grad():
            for emb, box, gm, _hw in dl_va:
                emb = emb.to(device); box = box.to(device)
                logits = decoder_forward(wrapper, emb, box)
                pr = (torch.sigmoid(logits) > 0.5).float()
                for k in range(pr.shape[0]):
                    inter = (pr[k] * gm[k].to(device)).sum().item()
                    dices.append((2 * inter + 1e-6) / (pr[k].sum().item() + gm[k].to(device).sum().item() + 1e-6))
        vd = float(np.mean(dices)) if dices else float("nan")
        print(f"[ep {ep:3d}] loss={tot/max(1,nb):.4f} val_dice256={vd:.4f} {time.time()-t0:.0f}s", flush=True)
        if vd > best:
            best = vd
            torch.save(dict(decoder=wrapper.model.mask_decoder.state_dict(), variant=args.variant,
                            epoch=ep, val_dice256=vd), out_dir / f"{args.variant}_decoder.pt")
    print(f"[done] best val_dice256={best:.4f} -> {out_dir}/{args.variant}_decoder.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
