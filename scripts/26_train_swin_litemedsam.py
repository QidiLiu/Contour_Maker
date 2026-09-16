"""在数据集上训练 Swin-LiteMedSAM 的 prompt encoder + mask decoder (编码器冻结)。

在线编码, 不落盘缓存 (4 个多尺度特征的缓存体积过大)。

官方 Swin-LiteMedSAM 权重仅在 Google Drive, 本机不可达 (见报告 §4.9), 因此:
  * Swin 编码器: 迁移 LiteMedSAM 中可对上的 18 个键, 其余随机初始化后**冻结**
  * prompt encoder + mask decoder: 用**本数据集的 YOLO26n 框提示**训练, 与其它方案同数据条件

    PYTHONPATH=src ./.conda/bin/python scripts/26_train_swin_litemedsam.py --epochs 8
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "LiteMedSAM"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "Swin_LiteMedSAM"))
from cm import metrics as M  # noqa: E402
from cm.config import ROOT, SPLITS, UNIFIED  # noqa: E402
from cm.litemedsam import IMG_SIZE, build_lite  # noqa: E402

OUT = ROOT / "weights" / "sam_finetuned"


def pick_box(boxes, gt_box, thr: float = 0.3):
    best, bi = None, 0.0
    for b in boxes:
        i = M.bbox_iou(tuple(map(int, b)), gt_box)
        if i > bi:
            best, bi = b, i
    return (best, bi) if bi >= thr else (None, bi)


class SwinDS(torch.utils.data.Dataset):
    """在线编码: 每个样本 = 一张图 (编码一次, 复用于该图所有目标)。"""

    def __init__(self, split: str, det_path: str, conf: float, imgsz: int, device,
                 wrapper, stride: int = 1):
        sp = pd.read_csv(SPLITS / "splits.csv")
        sp = sp[sp["split"] == split]
        self.items = list(sp.itertuples())[::stride]
        self.det_path, self.conf, self.imgsz = det_path, conf, imgsz
        self.device, self.wrapper = device, wrapper
        self._det = None

    @property
    def det(self):
        if self._det is None:
            from ultralytics import YOLO
            self._det = YOLO(self.det_path)
        return self._det

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row = self.items[i]
        img = cv2.imread(str(UNIFIED / row.image), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / row.mask), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            return None
        m = (gt > 127).astype(np.uint8)
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        comps = []
        for lab in range(1, ncomp):
            if stats[lab, cv2.CC_STAT_AREA] < 50:
                continue
            comp = (labels == lab).astype(np.uint8)
            bb = M.mask_bbox(comp)
            if bb:
                comps.append((comp, bb))
        if not comps:
            return None
        res = self.det.predict(str(UNIFIED / row.image), conf=self.conf, imgsz=self.imgsz,
                               verbose=False, device=str(self.device))[0]
        boxes = ([tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
                 if res.boxes is not None and len(res.boxes) else [])
        if not boxes:
            return None
        with torch.no_grad():
            e = self.wrapper.encode(img)
        emb = e["emb"][0].to(torch.float16).cpu()
        fs = [f[0].to(torch.float16).cpu() for f in e["fs"]]
        r = e["ratio"]; nh, nw = e["new_size"]
        # 每张图只取 1 个目标: Swin 版 decoder 要求 prompt 数与 image embedding 一一对应
        pairs = []
        for comp, gb in comps:
            b, _ = pick_box(boxes, gb)
            if b is not None:
                pairs.append((comp, b))
        if not pairs:
            return None
        comp, b = pairs[np.random.randint(len(pairs))]
        small = cv2.resize(comp, (nw, nh), interpolation=cv2.INTER_NEAREST)
        gm = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        gm[:nh, :nw] = small
        return (emb, fs, torch.tensor([v * r for v in b], dtype=torch.float32),
                torch.from_numpy(gm)[None])


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    emb = torch.stack([b[0] for b in batch])
    fs = [torch.stack([b[1][i] for b in batch]) for i in range(4)]
    boxes = torch.stack([b[2] for b in batch])      # (B,4)
    gts = torch.stack([b[3] for b in batch])        # (B,1,256,256)
    return emb, fs, boxes, gts


def dice_loss(logits, target, eps: float = 1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--val-stride", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = YOLO(args.det)
    w = build_lite("swin", device=str(device))
    model = w.model
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()):
        p.requires_grad = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 可训练参数 {n_tr/1e6:.2f}M (prompt+mask decoder); 编码器冻结", flush=True)

    ds_tr = SwinDS("train", args.det, args.conf, args.imgsz, device, w, stride=args.stride)
    ds_va = SwinDS("val", args.det, args.conf, args.imgsz, device, w, stride=args.val_stride)
    ds_tr._det = det        # 复用同一个 detector 实例
    ds_va._det = det
    # num_workers 必须为 0: Dataset 内部要做 CUDA 编码, fork 子进程无法重新初始化 CUDA
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=0,
                                        collate_fn=collate, drop_last=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=8, shuffle=False, num_workers=0,
                                        collate_fn=collate)
    print(f"[data] train 图={len(ds_tr)} val 图={len(ds_va)}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        t0 = time.time()
        for batch in dl_tr:
            if batch is None:
                continue
            emb, fs, box, gt = batch
            emb = emb.float().to(device)          # 编码以 fp16 缓存, 转回 fp32
            box, gt = box.to(device), gt.to(device)
            fs = [f.float().to(device) for f in fs]
            sparse, dense = model.prompt_encoder(points=None, boxes=box[:, None, :],
                                                 masks=None, tokens=None)
            low, _ = model.mask_decoder(fs, image_embeddings=emb,
                                        image_pe=model.prompt_encoder.get_dense_pe(),
                                        sparse_prompt_embeddings=sparse,
                                        dense_prompt_embeddings=dense, multimask_output=False)
            loss = F.binary_cross_entropy_with_logits(low, gt) + dice_loss(low, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()

        model.eval()
        dices = []
        with torch.no_grad():
            for batch in dl_va:
                if batch is None:
                    continue
                emb, fs, box, gt = batch
                emb = emb.float().to(device)
                box, gt = box.to(device), gt.to(device)
                fs = [f.float().to(device) for f in fs]
                sparse, dense = model.prompt_encoder(points=None, boxes=box[:, None, :],
                                                     masks=None, tokens=None)
                low, _ = model.mask_decoder(fs, image_embeddings=emb,
                                            image_pe=model.prompt_encoder.get_dense_pe(),
                                            sparse_prompt_embeddings=sparse,
                                            dense_prompt_embeddings=dense,
                                            multimask_output=False)
                pr = (torch.sigmoid(low) > 0.5).float()
                for k in range(pr.shape[0]):
                    inter = (pr[k] * gt[k]).sum().item()
                    dices.append((2 * inter + 1e-6) /
                                 (pr[k].sum().item() + gt[k].sum().item() + 1e-6))
        vd = float(np.mean(dices)) if dices else float("nan")
        print(f"[ep {ep:3d}] loss={tot/max(1,nb):.4f} val_dice256={vd:.4f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if vd > best:
            best = vd
            torch.save(dict(prompt_encoder=model.prompt_encoder.state_dict(),
                            mask_decoder=model.mask_decoder.state_dict(),
                            epoch=ep, val_dice256=vd), OUT / "swin_litemedsam_head.pt")
    print(f"[done] best val_dice256={best:.4f} -> {OUT/'swin_litemedsam_head.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
