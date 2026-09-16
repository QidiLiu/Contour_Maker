"""输入尺寸对比: A-box / B(YOLO26n-seg) / YOLO26n+EdgeSAM，在 512×512 与 256×256 下。

统一预处理: 图像 letterbox 到 S×S (保持长宽比, 右/下填充 114)，检测框随之缩放；
分割输出再映射回原图分辨率与 GT 比较 (原图分辨率固定 → 指标可横向对比, 也与
此前主结果口径一致)。同时输出在 S×S 分辨率下直接比较的版本。

指标: 分割 Dice/IoU/HD95/ASSD/边界F1/面积误差 + 目标级检测 P/R/F1 (IoU≥0.5)。
速度: 每张图端到端 wall-clock (含读图、预处理、检测、分割解码)，预热后统计。

    PYTHONPATH=src ./.conda/bin/python scripts/24_input_size_bench.py --sizes 512 256
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import geometry as G  # noqa: E402
from cm import metrics as M  # noqa: E402
from cm.config import IOU_MATCH_THRESHOLD, REPORTS, SPLITS, UNIFIED  # noqa: E402
from cm.rough import box_rough_contour  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ev", str(Path(__file__).resolve().parent / "07_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

PAD_VALUE = 114


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """等比缩放到 size×size 并填充。返回 (图, scale, pad_x, pad_y)。"""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size), PAD_VALUE, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, float(px), float(py), float(nw), float(nh)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def global_prf(preds: list[tuple[str, tuple]], gts: list[tuple[str, tuple]]) -> dict:
    if not preds or not gts:
        return dict(tp=0, fp=len(preds), fn=len(gts), precision=0.0, recall=0.0, f1=0.0)
    iou = np.zeros((len(preds), len(gts)), np.float32)
    for i, (_, pb) in enumerate(preds):
        for j, (_, gb) in enumerate(gts):
            iou[i, j] = M.bbox_iou(tuple(map(int, pb)), tuple(map(int, gb)))
    used_p, used_g, tp = set(), set(), 0
    for i, j in np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]:
        if i in used_p or j in used_g or iou[i, j] < IOU_MATCH_THRESHOLD:
            continue
        used_p.add(int(i)); used_g.add(int(j)); tp += 1
    P = tp / len(preds); R = tp / len(gts)
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(tp=tp, fp=len(preds) - tp, fn=len(gts) - tp, precision=P, recall=R, f1=F)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[512, 256])
    ap.add_argument("--refiner", default="runs/refiner_v3_A5_busi-ddti-tn3k/best.pt")
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--seg", default="runs/seg_yolo26n_sub/weights/best.pt")
    ap.add_argument("--sam-ckpt", default="weights/sam_finetuned")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=str(REPORTS / "input_size"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from ultralytics import YOLO
    from cm.sam_models import build_sam
    det = YOLO(args.det)
    seg = YOLO(args.seg)
    refiner, rcfg, _ = ev.load_refiner(Path(args.refiner), device)
    esam = build_sam("edgesam", device=str(device))
    ck = Path(args.sam_ckpt) / "edgesam_decoder.pt"
    if ck.exists():
        sd = torch.load(ck, map_location=device, weights_only=False)
        esam.model.mask_decoder.load_state_dict(sd["decoder"], strict=False)
        print(f"[bench] EdgeSAM 载入微调 decoder (val_dice256={sd.get('val_dice256'):.4f})")

    sp = pd.read_csv(SPLITS / "splits.csv")
    sub = pd.read_csv("data/yolo_subset/_subset_splits.csv")
    sub = sub[sub["split"] == "test"]
    sp = sp.merge(sub[["uid"]], on="uid").head(args.limit).reset_index(drop=True)

    cache = {}
    for row in sp.itertuples():
        im = cv2.imread(str(UNIFIED / row.image), cv2.IMREAD_GRAYSCALE)
        gm = cv2.imread(str(UNIFIED / row.mask), cv2.IMREAD_GRAYSCALE)
        if im is not None and gm is not None:
            cache[row.uid] = (im, gm)
    print(f"[bench] 测试图片 {len(cache)} 张, 尺寸 {args.sizes}", flush=True)

    def gt_comps(uid: str):
        """原图分辨率下的 GT 连通域 (mask, bbox)。"""
        gm = (cache[uid][1] > 127).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(gm, 8)
        out = []
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] < 50:
                continue
            comp = (labels == lab).astype(np.uint8)
            bb = M.mask_bbox(comp)
            if bb:
                out.append((comp, bb))
        return gm, out

    all_rows, speed_rows = [], []
    methods = ["A_box", "B_seg", "SAM_edgesam"]

    for size in args.sizes:
        S = size
        times = {m: [] for m in methods}
        records = {m: [] for m in methods}
        det_pred, det_gt = [], []

        def predict_all(img_orig: np.ndarray):
            """把原图交给模型内部预处理 (imgsz=S)。

            坐标映射**不使用**自行复现的 letterbox 几何, 而是用模型内部给出的
            `boxes.orig_shape` 反推缩放/填充 (由 letterbox 画布与网络输入尺寸的比值确定),
            这样与 ultralytics 的实现细节解耦。
            """
            H0, W0 = img_orig.shape[:2]
            out_masks: dict[str, list] = {m: [] for m in methods}

            # ---- 检测
            sync(device); t0 = time.perf_counter()
            res = det.predict(img_orig, conf=args.conf, imgsz=S, verbose=False, device=str(device))[0]
            sync(device); t_det = (time.perf_counter() - t0) * 1000.0
            # 注意: ultralytics 的 boxes.xyxy 已经是**原图坐标** (boxes.orig_shape == 原图尺寸),
            # 不能再做反向 letterbox 映射。
            boxes_o = ([tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
                       if res.boxes is not None and len(res.boxes) else [])

            # ---- A-box
            sync(device); t0 = time.perf_counter()
            for b in boxes_o:
                bb = box_rough_contour(img_orig, b)
                if not bb.ok:
                    continue
                pts0 = G.resample_closed(G.ensure_ccw(bb.contour), rcfg.n_points)
                pts = ev.refine_contour(refiner, rcfg, img_orig, b, pts0, device)
                mk = G.contour_to_mask(pts, (H0, W0))
                if mk.sum():
                    out_masks["A_box"].append(mk)
            sync(device); t_a = (time.perf_counter() - t0) * 1000.0

            # ---- B: YOLO26n-seg
            sync(device); t0 = time.perf_counter()
            rseg = seg.predict(img_orig, conf=args.conf, imgsz=S, verbose=False, device=str(device))[0]
            if getattr(rseg, "masks", None) is not None and rseg.masks is not None:
                md = rseg.masks.data.cpu().numpy()
                mh, mw = md.shape[1], md.shape[2]
                r = min(mh / H0, mw / W0)
                x0, y0 = int(round((mw - W0 * r) / 2)), int(round((mh - H0 * r) / 2))
                x1, y1 = int(round(x0 + W0 * r)), int(round(y0 + H0 * r))
                for k in range(md.shape[0]):
                    crop = np.ascontiguousarray((md[k] > 0.5).astype(np.uint8)[y0:y1, x0:x1])
                    if not crop.size:
                        continue
                    mk = cv2.resize(crop, (W0, H0), interpolation=cv2.INTER_NEAREST)
                    if mk.sum():
                        out_masks["B_seg"].append(mk)
            sync(device); t_b = (time.perf_counter() - t0) * 1000.0

            # ---- EdgeSAM
            sync(device); t0 = time.perf_counter()
            emb = esam.encode(img_orig) if boxes_o else None
            if emb is not None:
                for b in boxes_o:
                    m = esam.decode(emb, b, H0, W0)
                    m = cv2.resize(m.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)
                    if m.sum():
                        out_masks["SAM_edgesam"].append((m > 0).astype(np.uint8))
            sync(device); t_e = (time.perf_counter() - t0) * 1000.0

            return boxes_o, out_masks, dict(A_box_e2e=t_det + t_a, B_seg_e2e=t_b,
                                            SAM_edgesam_e2e=t_det + t_e, det=t_det)

        for uid in list(cache)[:args.warmup]:
            predict_all(cache[uid][0])
        sync(device)

        for n, row in enumerate(sp.itertuples(), 1):
            img_orig, _ = cache[row.uid]
            gt_mask = (cache[row.uid][1] > 127).astype(np.uint8)
            ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, 8)
            comps = [((labels == i).astype(np.uint8), M.mask_bbox((labels == i).astype(np.uint8)))
                     for i in range(1, ncomp) if stats[i, cv2.CC_STAT_AREA] >= 50]
            comps = [(c, b) for c, b in comps if b]
            for rep in range(args.repeats):
                boxes_o, out_masks, tm = predict_all(img_orig)
                if rep == 0:
                    for m in methods:
                        times[m].append(tm[f"{m}_e2e"])
                    det_gt.extend((row.uid, tuple(map(int, b))) for _, b in comps)
                    det_pred.extend((row.uid, tuple(map(int, b))) for b in boxes_o)
            for m in methods:
                preds = out_masks[m]
                if not preds or not comps:
                    records[m].append(dict(uid=row.uid, dataset=row.dataset, dice=0.0, iou=0.0,
                                           hd95=np.nan, assd=np.nan, bf1=0.0, area_err=np.nan))
                    continue
                iou = np.zeros((len(preds), len(comps)), np.float32)
                for i, pm in enumerate(preds):
                    for j, (cm, _) in enumerate(comps):
                        iou[i, j] = M.iou(pm, cm)
                used_p, used_g, matched = set(), set(), 0
                for i, j in np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]:
                    if i in used_p or j in used_g or iou[i, j] < IOU_MATCH_THRESHOLD:
                        continue
                    used_p.add(int(i)); used_g.add(int(j)); matched += 1
                    rec = dict(uid=row.uid, dataset=row.dataset)
                    rec.update(M.evaluate_pair(preds[int(i)], comps[int(j)][0]))
                    records[m].append(rec)
                if matched == 0:
                    records[m].append(dict(uid=row.uid, dataset=row.dataset, dice=0.0, iou=0.0,
                                           hd95=np.nan, assd=np.nan, bf1=0.0, area_err=np.nan))
            if n % 50 == 0:
                print(f"  [{S}] {n}/{len(sp)}", flush=True)

        prf = global_prf(det_pred, det_gt)
        print(f"[{S}] 检测: TP={prf['tp']} FP={prf['fp']} FN={prf['fn']} "
              f"P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}", flush=True)
        for m in methods:
            arr = np.array(times[m]) * 1.0
            speed_rows.append(dict(size=S, method=m, n_images=len(arr),
                                   end2end_ms_mean=arr.mean(), end2end_ms_median=np.median(arr),
                                   end2end_ms_std=arr.std(), fps=1000.0 / arr.mean(),
                                   **{f"det_{k}": v for k, v in prf.items()}))
            df = pd.DataFrame(records[m])
            all_rows.append(dict(size=S, method=m, n=len(df), dice=df["dice"].mean(),
                                 dice_std=df["dice"].std(), iou=df["iou"].mean(),
                                 hd95=df["hd95"].mean(), assd=df["assd"].mean(),
                                 bf1=df["bf1"].mean(), area_err=df["area_err"].mean(),
                                 fail=float((df["dice"] < 0.5).mean()),
                                 **{f"det_{k}": v for k, v in prf.items()}))
            df.to_csv(out / f"per_image_{m}_{S}.csv", index=False)
        pd.DataFrame(speed_rows).to_csv(out / "speed.csv", index=False)
        pd.DataFrame(all_rows).to_csv(out / "accuracy_speed.csv", index=False)
        print(f"[{S}] 完成", flush=True)

    t = pd.DataFrame(all_rows)
    s = pd.DataFrame(speed_rows)
    print("\n=== 输入尺寸 × 方案 (分割指标, 原图分辨率下评估) ===")
    print(t[["size", "method", "n", "dice", "iou", "hd95", "assd", "bf1", "fail"]].round(4).to_string(index=False))
    print("\n=== 检测指标 (目标级, IoU≥0.5) ===")
    print(t[["size", "method", "det_tp", "det_fp", "det_fn", "det_precision", "det_recall", "det_f1"]]
          .round(4).to_string(index=False))
    print("\n=== 端到端速度 ===")
    print(s.round(2).to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
