"""训练 contour-refiner (方案 A 的轮廓精细化模型)。

用法:
    PYTHONPATH=src ./.conda/bin/python scripts/03_train_refiner.py --level 5 --tag A5
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import geometry as G  # noqa: E402
from cm import metrics as M  # noqa: E402
from cm import roi as R  # noqa: E402
from cm.config import RUNS, SPLITS, UNIFIED, AugConfig, RefinerConfig  # noqa: E402
from cm.model import build_loss, build_refiner, count_params  # noqa: E402

N_WORKERS_DEFAULT = 4


# ---------------------------------------------------------------- dataset
class RefinerDataset(Dataset):
    """在线生成 (64x32x32 ROI -> 64 点) 样本; 训练时每 epoch 重新采样粗糙轮廓与增强。"""

    polarity = "dark"      # 超声低回声病灶: 目标为暗区

    def __init__(self, df: pd.DataFrame, rcfg: RefinerConfig, acfg: AugConfig,
                 train: bool, samples_per_target: int = 1, cache: bool = True,
                 jitter: bool | None = None, with_shape: bool = False):
        self.df = df.reset_index(drop=True)
        self.rcfg, self.acfg = rcfg, acfg
        self.train = train
        self.spt = samples_per_target if train else 1
        self.jitter = train if jitter is None else jitter
        self.cache = cache
        self.with_shape = with_shape
        self._imgs: dict[str, np.ndarray] = {}
        self._gts: dict[str, np.ndarray] = {}
        self.targets: list[tuple[int, int]] = []
        self._gt_info: dict[int, list[dict]] = {}
        for i, row in self.df.iterrows():
            mask = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            m = (mask > 127).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            infos = []
            for lab in range(1, n):
                comp = (labels == lab).astype(np.uint8)
                if comp.sum() < 50:
                    continue
                bbox = R.bbox_from_mask(comp)
                if bbox is None:
                    continue
                infos.append(dict(bbox=bbox, area=int(comp.sum()), label=int(lab)))
                self.targets.append((i, len(infos) - 1))
            self._gt_info[i] = infos

    def __len__(self) -> int:
        return len(self.targets) * self.spt

    def _load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.df.iloc[i]
        uid = row["uid"]
        if self.cache and uid in self._imgs:
            return self._imgs[uid], self._gts[uid]
        img = cv2.imread(str(UNIFIED / row["image"]), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if self.cache:
            self._imgs[uid] = img
            self._gts[uid] = gt
        return img, gt

    def __getitem__(self, idx: int):
        row_i, tgt_i = self.targets[idx // self.spt]
        img, gt = self._load(row_i)
        info = self._gt_info[row_i][tgt_i]
        x1, y1, x2, y2 = [int(round(v)) for v in info["bbox"]]
        x1, y1 = max(0, x1 - 2), max(0, y1 - 2)
        x2, y2 = min(gt.shape[1], x2 + 2), min(gt.shape[0], y2 + 2)
        sub = np.zeros_like(gt, np.uint8)
        m = (gt > 127).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        lab = int(info["label"])
        if n > lab:
            sub = (labels == lab).astype(np.uint8)
        else:
            sub[y1:y2, x1:x2] = (gt[y1:y2, x1:x2] > 127).astype(np.uint8)
        spec = R.TargetSpec(image=img, gt_mask=sub, bbox=(x1, y1, x2, y2), polarity=self.polarity)
        rng = np.random.default_rng((np.random.randint(0, 2 ** 31) + idx * 7919) % (2 ** 32))
        for _ in range(4):
            s = R.build_sample(spec, self.rcfg, self.acfg, rng, jitter=self.jitter)
            if s is not None:
                return (s, img.shape) if self.with_shape else s
        return None


def collate(batch):
    items = [b for b in batch if b is not None]
    if not items:
        return None
    # 兼容 (Sample, shape) 与 Sample 两种返回形式
    samples = [it[0] if isinstance(it, tuple) else it for it in items]
    patches = torch.from_numpy(np.stack([s.patches for s in samples])).float()
    rough = torch.from_numpy(np.stack([s.rough_norm for s in samples])).float()
    target = torch.from_numpy(np.stack([s.target_norm for s in samples])).float()
    aux = torch.from_numpy(np.stack([s.aux for s in samples])).float()
    centers = torch.from_numpy(np.stack([[f.center_norm for f in s.frames] for s in samples])).float()
    shapes = [s.shape for s in samples]
    return patches, rough, target, aux, centers, shapes, samples


# ---------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, loader, device, lambda_smooth: float = 0.0) -> dict:
    model.eval()
    dices, ious, rough_dices, losses = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        patches, rough, target, aux, centers, shapes, samples = batch
        patches, rough = patches.to(device), rough.to(device)
        target, aux, centers = target.to(device), aux.to(device), centers.to(device)
        pred = model(patches, rough, aux, centers)
        loss, _ = build_loss(pred, target, "smooth_l1", lambda_smooth)
        losses.append(float(loss))
        pred_np = pred.float().cpu().numpy()
        for k, s in enumerate(samples):
            shape = shapes[k]
            gm = G.contour_to_mask(s.meta["exact_global"], shape)
            pred_mask = G.contour_to_mask(R.frame_from_norm(pred_np[k], s.frames), shape)
            rough_mask = G.contour_to_mask(s.rough_global, shape)
            dices.append(M.dice(pred_mask, gm))
            ious.append(M.iou(pred_mask, gm))
            rough_dices.append(M.dice(rough_mask, gm))
    return dict(val_loss=float(np.mean(losses)) if losses else float("nan"),
                val_dice=float(np.mean(dices)) if dices else float("nan"),
                val_iou=float(np.mean(ious)) if ious else float("nan"),
                val_rough_dice=float(np.mean(rough_dices)) if rough_dices else float("nan"),
                n=int(len(dices)))


# ---------------------------------------------------------------- cached dataset
def _f16(b) -> np.ndarray:
    return np.frombuffer(b, np.float16).astype(np.float32)


class CachedRefinerDataset(Dataset):
    """从 scripts/03b_cache_samples.py 生成的缓存读取样本 (大幅提升训练吞吐)。"""

    def __init__(self, cache_dir: Path, split: str, n_points: int = 64, roi_size: int = 32,
                 datasets: list[str] | None = None):
        self.dir = Path(cache_dir)
        self.mos = (self.dir / "mosaics.bin").read_bytes()
        df = pd.read_parquet(self.dir / "samples.parquet")
        if datasets:
            df = df[df["dataset"].isin(datasets)]
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.n_points = n_points
        self.roi_size = roi_size
        self.side = int(np.ceil(np.sqrt(n_points)))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        tile = self.roi_size
        side = self.side
        img = np.frombuffer(self.mos, np.uint8, int(r["length"]), int(r["offset"]))
        img = img.reshape(side * tile, side * tile)
        patches = np.zeros((self.n_points, tile, tile), np.float32)
        for i in range(self.n_points):
            rr, cc = divmod(i, side)
            patches[i] = img[rr * tile:(rr + 1) * tile, cc * tile:(cc + 1) * tile].astype(np.float32) / 255.0 * 8.0 - 4.0
        return dict(patches=patches,
                    rough_norm=_f16(r["rough"]).reshape(self.n_points, 2),
                    target_norm=_f16(r["target"]).reshape(self.n_points, 2),
                    aux=_f16(r["aux"]).reshape(self.n_points, 3),
                    centers=_f16(r["centers"]).reshape(self.n_points, 2),
                    windows=_f16(r["windows"]).reshape(self.n_points),
                    bbox=_f16(r["bbox"]).reshape(4),
                    exact=_f16(r["exact"]).reshape(self.n_points, 2),
                    rough_global=_f16(r["roughg"]).reshape(self.n_points, 2),
                    shape=(int(r["shape_h"]), int(r["shape_w"])),
                    uid=str(r["uid"]), dataset=str(r["dataset"]))


def collate_cached(batch):
    patches = torch.from_numpy(np.stack([b["patches"] for b in batch])).float()
    rough = torch.from_numpy(np.stack([b["rough_norm"] for b in batch])).float()
    target = torch.from_numpy(np.stack([b["target_norm"] for b in batch])).float()
    aux = torch.from_numpy(np.stack([b["aux"] for b in batch])).float()
    centers = torch.from_numpy(np.stack([b["centers"] for b in batch])).float()
    return patches, rough, target, aux, centers, batch


def _frames_from_cache(b: dict) -> list[R.FrameInfo]:
    """由缓存中的 windows/centers 还原 FrameInfo (供 mask 重建)。"""
    H, W = b["shape"]
    out = []
    for c, w in zip(b["centers"], b["windows"]):
        out.append(R.FrameInfo(scale=float(w), center=(c * np.array([W, H], np.float32)).astype(np.float32),
                               patch_px=int(round(float(w))), center_norm=c.astype(np.float32)))
    return out


@torch.no_grad()
def validate_cached(model, loader, device, sample_fn) -> dict:
    model.eval()
    dices, ious, rough_dices, losses = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        patches, rough, target, aux, centers, items = batch
        patches, rough = patches.to(device), rough.to(device)
        target, aux, centers = target.to(device), aux.to(device), centers.to(device)
        pred = model(patches, rough, aux, centers)
        loss, _ = build_loss(pred, target)
        losses.append(float(loss))
        pred_np = pred.float().cpu().numpy()
        for k, b in enumerate(items):
            gm = G.contour_to_mask(b["exact"], b["shape"])
            dices.append(M.dice(sample_fn(pred_np[k], b), gm))
            ious.append(M.iou(sample_fn(pred_np[k], b), gm))
            rough_dices.append(M.dice(sample_fn(b["rough_norm"], b), gm))
    return dict(val_loss=float(np.mean(losses)) if losses else float("nan"),
                val_dice=float(np.mean(dices)) if dices else float("nan"),
                val_iou=float(np.mean(ious)) if ious else float("nan"),
                val_rough_dice=float(np.mean(rough_dices)) if rough_dices else float("nan"),
                n=int(len(dices)))


def cached_mask(norm: np.ndarray, b: dict) -> np.ndarray:
    frames = _frames_from_cache(b)
    return G.contour_to_mask(R.frame_from_norm(norm, frames), b["shape"])


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5, help="数据增强等级 A1..A5 (0=无增强)")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--window-ratio", type=float, default=None)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--roi-size", type=int, default=None)
    ap.add_argument("--feat-dim", type=int, default=None)
    ap.add_argument("--tx-layers", type=int, default=None)
    ap.add_argument("--lambda-smooth", type=float, default=0.0)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--samples-per-target", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=N_WORKERS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--polarity", default="dark", choices=["dark", "bright", "auto"])
    ap.add_argument("--cache-dir", default=None,
                    help="样本缓存目录 (由 scripts/03b_cache_samples.py 生成); 指定后使用缓存训练")
    args = ap.parse_args()

    rcfg = RefinerConfig()
    for k, v in [("epochs", args.epochs), ("batch_size", args.batch_size), ("lr", args.lr),
                 ("window_ratio", args.window_ratio), ("n_points", args.n_points),
                 ("roi_size", args.roi_size), ("feat_dim", args.feat_dim),
                 ("n_tx_layers", args.tx_layers),
                 ("samples_per_target", args.samples_per_target)]:
        if v is not None:
            setattr(rcfg, k, v)
    rcfg.seed = args.seed

    acfg = AugConfig(level=args.level)
    tag = args.tag or f"A{args.level}"
    if args.datasets:
        tag += "_" + "-".join(sorted(args.datasets))
    out_dir = RUNS / f"refiner_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    RefinerDataset.polarity = args.polarity

    sp = pd.read_csv(SPLITS / "splits.csv")
    if args.datasets:
        sp = sp[sp["dataset"].isin(args.datasets)]
    tr, va = sp[sp["split"] == "train"], sp[sp["split"] == "val"]
    print(f"[data] train={len(tr)} val={len(va)} datasets={sorted(sp['dataset'].unique())}", flush=True)

    if args.cache_dir:
        ds_tr = CachedRefinerDataset(args.cache_dir, "train", rcfg.n_points, rcfg.roi_size, args.datasets)
        ds_va = CachedRefinerDataset(args.cache_dir, "val", rcfg.n_points, rcfg.roi_size, args.datasets)
        collate_fn = collate_cached
        print(f"[data] 使用缓存 {args.cache_dir}: train={len(ds_tr)} val={len(ds_va)}", flush=True)
    else:
        ds_tr = RefinerDataset(tr, rcfg, acfg, train=True, samples_per_target=rcfg.samples_per_target,
                               jitter=True)
        ds_va = RefinerDataset(va, rcfg, acfg, train=False, jitter=True, with_shape=True)
        collate_fn = collate
        print(f"[data] train targets={len(ds_tr.targets)} val targets={len(ds_va.targets)}", flush=True)

    dl_tr = DataLoader(ds_tr, batch_size=rcfg.batch_size, shuffle=True, num_workers=args.num_workers,
                       collate_fn=collate_fn, drop_last=True, persistent_workers=args.num_workers > 0)
    dl_va = DataLoader(ds_va, batch_size=rcfg.batch_size, shuffle=False,
                       num_workers=max(1, args.num_workers // 2), collate_fn=collate_fn,
                       persistent_workers=args.num_workers > 0)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = build_refiner(rcfg).to(device)
    print(f"[model] params={count_params(model)/1e3:.1f}K feat_dim={rcfg.feat_dim} "
          f"points={rcfg.n_points} roi={rcfg.roi_size} window_ratio={rcfg.window_ratio} "
          f"device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=rcfg.lr, weight_decay=rcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rcfg.epochs, eta_min=rcfg.lr * 0.02)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, hist = -1.0, []
    eval_every = max(1, min(10, rcfg.epochs // 10))
    for ep in range(1, rcfg.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for batch in dl_tr:
            if batch is None:
                continue
            patches, rough, target, aux, centers = batch[:5]
            patches = patches.to(device, non_blocking=True)
            rough = rough.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            centers = centers.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(patches, rough, aux, centers)
                loss, _ = build_loss(pred, target, rcfg.loss, args.lambda_smooth)
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

        if ep % eval_every == 0 or ep == 1 or ep == rcfg.epochs:
            if args.cache_dir:
                res = validate_cached(model, dl_va, device, cached_mask)
            else:
                res = validate(model, dl_va, device, args.lambda_smooth)
            rec = dict(epoch=ep, train_loss=tr_loss, **res, lr=sched.get_last_lr()[0],
                       sec=round(time.time() - t0, 1))
            hist.append(rec)
            print(f"[ep {ep:3d}] train_loss={tr_loss:.5f} val_loss={res['val_loss']:.5f} "
                  f"dice={res['val_dice']:.4f} (rough {res['val_rough_dice']:.4f}) "
                  f"iou={res['val_iou']:.4f} n={res['n']} {rec['sec']}s", flush=True)
            if res["val_dice"] > best:
                best = res["val_dice"]
                torch.save(dict(model=model.state_dict(), rcfg=vars(rcfg), acfg=vars(acfg),
                                level=args.level, tag=tag, epoch=ep, val_dice=best,
                                polarity=args.polarity), out_dir / "best.pt")
        else:
            print(f"[ep {ep:3d}] train_loss={tr_loss:.5f} {time.time()-t0:.1f}s", flush=True)

    torch.save(dict(model=model.state_dict(), rcfg=vars(rcfg), acfg=vars(acfg), level=args.level,
                    tag=tag, epoch=rcfg.epochs, val_dice=best, polarity=args.polarity),
               out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    (out_dir / "config.json").write_text(json.dumps(dict(rcfg=vars(rcfg), acfg=vars(acfg),
                                                         level=args.level, tag=tag), indent=2))
    print(f"[done] best val dice={best:.4f} -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
