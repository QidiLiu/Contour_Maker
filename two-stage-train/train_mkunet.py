"""训练 MK-UNet（两阶段的第二级）。

输入是 YOLO26n 检测框裁剪出的 ROI（已 resize 到 --roi-size），
输出是该 ROI 内的前景/背景二值掩码。

用法:
    python train_mkunet.py --data ../data/roi_cache/train_gt \
        --val-data ../data/roi_cache/val_gt \
        --epochs 60 --batch 16 --name mkunet_yolo26n
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    MKUNET_BATCH,
    MKUNET_EPOCHS,
    MKUNET_LR,
    MKUNET_ROI_SIZE,
    MKUNET_VARIANT,
    MKUNET_WEIGHT_DECAY,
    RUNS,
)
from common.metrics import dice as dice_metric  # noqa: E402
from common.mkunet import build_mkunet, count_params  # noqa: E402


# ---------------------------------------------------------------- dataset
class ROIDataset(Dataset):
    """读取 build_roi_dataset.py 产出的 ROI 图像/掩码对。"""

    def __init__(self, root: Path, split: str | None = None, roi_size: int = 256,
                 augment: bool = False):
        self.root = Path(root)
        idx = self.root / "index.csv"
        if not idx.exists():
            raise FileNotFoundError(f"未找到 {idx}")
        self.df = pd.read_csv(idx)
        if split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.roi_size = roi_size
        self.augment = augment
        self.files = self.df["file"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, i: int):
        r = self.df.iloc[i]
        split = r["split"]
        img = cv2.imread(str(self.root / split / r["file"]), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(self.root / split / r["mask_file"]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            return None
        if img.shape[:2] != (self.roi_size, self.roi_size):
            img = cv2.resize(img, (self.roi_size, self.roi_size),
                             interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (self.roi_size, self.roi_size),
                             interpolation=cv2.INTER_NEAREST)
        return img, (msk > 127).astype(np.uint8)

    def __getitem__(self, i: int):
        got = self._load(i)
        if got is None:
            return None
        img, msk = got

        if self.augment:
            img, msk = _augment(img, msk)

        # (1, H, W) float [0,1]
        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None]
        y = torch.from_numpy(msk.astype(np.float32))[None]
        return x, y


def _augment(img: np.ndarray, msk: np.ndarray):
    """轻量增强：翻转 + 亮度/对比度 + 噪声。超声为灰度，关闭几何形变。"""
    if np.random.rand() < 0.5:
        img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
    if np.random.rand() < 0.5:
        img, msk = img[::-1, :].copy(), msk[::-1, :].copy()
    if np.random.rand() < 0.5:
        alpha = 1.0 + np.random.uniform(-0.25, 0.25)
        beta = np.random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.2:
        sigma = np.random.uniform(2, 8)
        img = np.clip(img.astype(np.float32) + np.random.normal(0, sigma, img.shape),
                      0, 255).astype(np.uint8)
    return img, msk


def collate(batch):
    items = [b for b in batch if b is not None]
    if not items:
        return None
    xs = torch.stack([b[0] for b in items])
    ys = torch.stack([b[1] for b in items])
    return xs, ys


# ---------------------------------------------------------------- loss
def dice_loss(pred, target, eps: float = 1.0):
    p = torch.sigmoid(pred)
    p = p.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def build_criterion(bce_w: float = 0.5, dice_w: float = 0.5):
    bce = nn.BCEWithLogitsLoss()

    def fn(pred, target):
        return bce_w * bce(pred, target) + dice_w * dice_loss(pred, target)

    return fn


# ---------------------------------------------------------------- validate
@torch.no_grad()
def validate(model, loader, device, amp: bool) -> dict:
    model.eval()
    dices, losses = [], []
    crit = build_criterion()
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logit = model(x)
            loss = crit(logit, y)
        losses.append(float(loss))
        pm = (torch.sigmoid(logit.float()) > 0.5).to(torch.uint8)
        for k in range(pm.shape[0]):
            a = pm[k, 0].cpu().numpy()
            b = y[k, 0].cpu().numpy().astype(np.uint8)
            dices.append(dice_metric(a, b))
    return dict(
        val_loss=float(np.mean(losses)) if losses else float("nan"),
        val_dice=float(np.mean(dices)) if dices else float("nan"),
        n=len(dices),
    )


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="训练 MK-UNet")
    ap.add_argument("--data", required=True, help="训练 ROI 数据目录")
    ap.add_argument("--val-data", default=None, help="验证 ROI 数据目录，默认同 --data")
    ap.add_argument("--variant", default=MKUNET_VARIANT,
                    choices=["MK_UNet", "MK_UNet_S", "MK_UNet_T"])
    ap.add_argument("--roi-size", type=int, default=MKUNET_ROI_SIZE)
    ap.add_argument("--epochs", type=int, default=MKUNET_EPOCHS)
    ap.add_argument("--batch", type=int, default=MKUNET_BATCH)
    ap.add_argument("--lr", type=float, default=MKUNET_LR)
    ap.add_argument("--weight-decay", type=float, default=MKUNET_WEIGHT_DECAY)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="mkunet_yolo26n")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available()
                          else "cpu")
    amp = device.type == "cuda"

    val_root = Path(args.val_data) if args.val_data else Path(args.data)
    ds_tr = ROIDataset(args.data, "train", args.roi_size, augment=True)
    ds_va = ROIDataset(val_root, "val", args.roi_size, augment=False)
    print(f"[data] train={len(ds_tr)} val={len(ds_va)} roi={args.roi_size} "
          f"device={device}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, collate_fn=collate,
                       drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=max(1, args.workers // 2), collate_fn=collate)

    model = build_mkunet(args.variant, num_classes=1, in_channels=1).to(device)
    print(f"[model] {args.variant} params={count_params(model)/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = build_criterion()

    start_ep = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[resume] from {args.resume} (epoch {ck.get('epoch')})")
        start_ep = int(ck.get("epoch", 0)) + 1

    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    hist = []
    eval_every = max(1, min(5, args.epochs // 10))

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for batch in dl_tr:
            if batch is None:
                continue
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logit = model(x)
                loss = crit(logit, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach())
            nb += 1
        sched.step()
        tr_loss = run_loss / max(1, nb)

        if ep % eval_every == 0 or ep == 1 or ep == args.epochs:
            res = validate(model, dl_va, device, amp)
            rec = dict(epoch=ep, train_loss=tr_loss, **res,
                       lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1))
            hist.append(rec)
            print(f"[ep {ep:3d}] train_loss={tr_loss:.5f} val_loss={res['val_loss']:.5f} "
                  f"val_dice={res['val_dice']:.4f} n={res['n']} {rec['sec']}s", flush=True)
            if res["val_dice"] > best:
                best = res["val_dice"]
                torch.save(dict(model=model.state_dict(),
                                variant=args.variant, roi_size=args.roi_size,
                                epoch=ep, val_dice=best),
                           out_dir / "best.pt")
        else:
            print(f"[ep {ep:3d}] train_loss={tr_loss:.5f} "
                  f"{time.time()-t0:.1f}s", flush=True)

    torch.save(dict(model=model.state_dict(), variant=args.variant,
                    roi_size=args.roi_size, epoch=args.epochs, val_dice=best),
               out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[done] best val_dice={best:.4f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())