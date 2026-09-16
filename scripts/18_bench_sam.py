"""统一流水线评测: 精度 + 平均推理速度。

四条流水线 (同一测试集/同一划分):
  1. A-box      : YOLO26n 检测框 -> 框矩形初始轮廓 -> contour-refiner
  2. B          : YOLO26n-seg 端到端实例分割
  3. YOLO26n + EdgeSAM       (box prompt)
  4. YOLO26n + EfficientSAM  (box prompt, ViT-T / ViT-S)

速度口径: 每张图端到端 wall-clock (含图像读取、检测、预处理、分割解码), GPU 同步计时,
先预热再统计 mean / median / std。

    PYTHONPATH=src ./.conda/bin/python scripts/18_bench_sam.py --limit 120 --repeats 3
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
from cm import roi as R  # noqa: E402
from cm.config import IOU_MATCH_THRESHOLD, REPORTS, SPLITS, UNIFIED  # noqa: E402
from cm.rough import box_rough_contour, otsu_adaptive_rough_contour  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ev", str(Path(__file__).resolve().parent / "07_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _edge_score(img, mask) -> float:
    """候选初始轮廓的边界证据打分: 边界梯度均值 - 内部灰度标准差。"""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 20:
        return -1e9
    band = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    e = float(g[band > 0].mean()) if (band > 0).any() else 0.0
    return e - 0.5 * float(img[m > 0].std())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def match_pairs(preds, gts, thr=IOU_MATCH_THRESHOLD):
    if not preds or not gts:
        return [], len(gts), len(preds)
    iou = np.zeros((len(preds), len(gts)), np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou[i, j] = M.iou(p, g)
    used_p, used_g, pairs = set(), set(), []
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for i, j in order:
        if i in used_p or j in used_g or iou[i, j] < thr:
            continue
        used_p.add(int(i)); used_g.add(int(j))
        pairs.append((preds[int(i)], gts[int(j)], float(iou[i, j])))
    return pairs, len(gts) - len(used_g), len(preds) - len(used_p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", default="runs/refiner_v3_A5_busi-ddti-tn3k/best.pt")
    ap.add_argument("--det", default="runs/det_yolo26n_sub/weights/best.pt")
    ap.add_argument("--seg", default="runs/seg_yolo26n_sub/weights/best.pt")
    ap.add_argument("--sam", nargs="*", default=["edgesam", "efficientvit-t"],
                    choices=["edgesam", "efficientvit-t", "efficientvit-s"])
    ap.add_argument("--sam-ckpt", default=None, help="可选: 微调后的 SAM 权重目录")
    ap.add_argument("--split", default="test")
    ap.add_argument("--uids-file", default="data/yolo_subset/_subset_splits.csv")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 张 (速度测试建议 100+)")
    ap.add_argument("--repeats", type=int, default=1, help="速度测试重复轮数 (取每张图平均)")
    ap.add_argument("--warmup", type=int, default=5, help="正式计时前的预热图片数")
    ap.add_argument("--a-variants", default="all",
                    help="方案 A 的初始化策略: all 或逗号分隔 (A_box,A_otsu,A_otsu_fb,A_select)")
    ap.add_argument("--no-accuracy", action="store_true", help="只测速度")
    ap.add_argument("--out", default=str(REPORTS / "bench_sam"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device} limit={args.limit or '全部'} repeats={args.repeats}", flush=True)

    from ultralytics import YOLO
    det = YOLO(args.det)
    seg = YOLO(args.seg)
    refiner, rcfg, _ = ev.load_refiner(Path(args.refiner), device)

    sams = {}
    for v in args.sam:
        from cm.sam_models import build_sam
        w = build_sam(v, device=str(device))
        if args.sam_ckpt:
            cand = Path(args.sam_ckpt) / f"{v}_decoder.pt"
            if cand.exists():
                sd = torch.load(cand, map_location=device, weights_only=False)
                missing, unexpected = w.model.mask_decoder.load_state_dict(sd["decoder"], strict=False)
                print(f"[bench] {v} 载入微调 decoder ({cand.name}) "
                      f"epoch={sd.get('epoch')} val_dice256={sd.get('val_dice256'):.4f} "
                      f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            else:
                print(f"[bench] {v}: 未找到 {cand}, 使用零样本权重", flush=True)
        sams[v] = w
        print(f"[bench] 加载 {sams[v].name} ({sams[v].n_params_m:.2f}M)", flush=True)

    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["split"] == args.split]
    if args.uids_file:
        keep = set(pd.read_csv(args.uids_file)["uid"].astype(str))
        sp = sp[sp["uid"].astype(str).isin(keep)]
    sp = sp.reset_index(drop=True)
    if args.limit:
        sp = sp.head(args.limit)
    print(f"[bench] 测试图片 {len(sp)} 张", flush=True)

    a_variants = [v for v in ("A_box", "A_otsu", "A_otsu_fb", "A_select")
                  if args.a_variants == "all" or v in args.a_variants.split(",")]
    methods = a_variants + ["B_seg"] + [f"SAM_{v}" for v in args.sam]

    # 预读图片到内存, 避免磁盘 I/O 干扰速度测量
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in sp.itertuples():
        im = cv2.imread(str(UNIFIED / row.image), cv2.IMREAD_GRAYSCALE)
        gm = cv2.imread(str(UNIFIED / row.mask), cv2.IMREAD_GRAYSCALE)
        if im is not None and gm is not None:
            cache[row.uid] = (im, gm)
    print(f"[bench] 已缓存 {len(cache)} 张图", flush=True)

    def run_all(img: np.ndarray) -> dict[str, tuple[list[np.ndarray], float, float]]:
        """对单张图跑全部方案, 返回 {method: (preds, t_detect, t_segment)}。"""
        H, W = img.shape[:2]
        sync(device); t0 = time.perf_counter()
        res = det.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False, device=str(device))[0]
        sync(device); t_det = time.perf_counter() - t0
        boxes = ([tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]
                 if res.boxes is not None and len(res.boxes) else [])
        out: dict[str, tuple[list[np.ndarray], float, float]] = {}

        # ---- 方案 A 的三种初始化策略 (共用同一个 refiner) ----
        def refine(init_pts, b):
            p2 = ev.refine_contour(refiner, rcfg, img, b, init_pts, device)
            return G.contour_to_mask(p2, (H, W))

        sync(device); t0 = time.perf_counter()
        p_box, p_otsu, p_fb, p_sel = [], [], [], []
        for b in boxes:
            bb = box_rough_contour(img, b)
            box_pts = (G.resample_closed(G.ensure_ccw(bb.contour), rcfg.n_points) if bb.ok else None)
            rr = otsu_adaptive_rough_contour(img, b, polarity="dark")
            otsu_pts = (G.resample_closed(G.ensure_ccw(rr.contour), rcfg.n_points)
                        if (rr.ok and len(rr.contour) >= 3) else None)
            # 纯框
            if box_pts is not None:
                mk = refine(box_pts, b)
                if mk.sum():
                    p_box.append(mk)
            # 纯 Otsu (失败则跳过)
            if otsu_pts is not None:
                mk = refine(otsu_pts, b)
                if mk.sum():
                    p_otsu.append(mk)
            # Otsu + 框回退 (训练/推理分布一致: 有 Otsu 用 Otsu, 否则用框)
            init_fb = otsu_pts if otsu_pts is not None else box_pts
            if init_fb is not None:
                mk = refine(init_fb, b)
                if mk.sum():
                    p_fb.append(mk)
            # 两候选择优 (边界梯度证据打分, 只精细化胜者)
            if otsu_pts is not None and box_pts is not None:
                s_o = _edge_score(img, G.contour_to_mask(otsu_pts, (H, W)))
                s_b = _edge_score(img, G.contour_to_mask(box_pts, (H, W)))
                init_sel = otsu_pts if s_o >= s_b else box_pts
            else:
                init_sel = otsu_pts if otsu_pts is not None else box_pts
            if init_sel is not None:
                mk = refine(init_sel, b)
                if mk.sum():
                    p_sel.append(mk)
        sync(device); t_seg = time.perf_counter() - t0
        n_a = 4
        out["A_box"] = (p_box, t_det, t_seg / n_a)
        out["A_otsu"] = (p_otsu, t_det, t_seg / n_a)
        out["A_otsu_fb"] = (p_fb, t_det, t_seg / n_a)
        out["A_select"] = (p_sel, t_det, t_seg / n_a)

        # B: YOLO26n-seg (检测+分割一次前向)
        sync(device); t0 = time.perf_counter()
        rseg = seg.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False, device=str(device))[0]
        preds = []
        if getattr(rseg, "masks", None) is not None and rseg.masks is not None:
            for poly in rseg.masks.xy:
                if poly is None or len(poly) < 3:
                    continue
                mk = G.contour_to_mask(np.asarray(poly, np.float32), (H, W))
                if mk.sum():
                    preds.append(mk)
        sync(device); t_tot = time.perf_counter() - t0
        out["B_seg"] = (preds, 0.0, t_tot)

        # SAM 系列 (共用检测框提示)
        for v, w in sams.items():
            sync(device); t0 = time.perf_counter()
            emb = w.encode(img) if boxes else None
            sync(device); t_enc = time.perf_counter() - t0
            preds = []
            if emb is not None:
                for b in boxes:
                    m = w.decode(emb, b, H, W)
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                    if m.sum():
                        preds.append((m > 0).astype(np.uint8))
            sync(device); t_tot2 = time.perf_counter() - t0
            out[f"SAM_{v}"] = (preds, t_det + t_enc, t_tot2 - t_enc)
        return out

    warm = [r.uid for r in sp.itertuples()][: max(0, args.warmup)]
    for uid in warm:
        run_all(cache[uid][0])
    sync(device)
    times = {m: [] for m in methods}
    comp_times = {m: [] for m in methods}
    records = []
    print(f"[bench] 预热完成 ({len(warm)} 张), 开始正式计时", flush=True)

    for n, row in enumerate(sp.itertuples(), 1):
        img_path = UNIFIED / row.image
        if row.uid not in cache:
            continue
        image, gt = cache[row.uid]
        for rep in range(args.repeats):
            per_method = run_all(image)

            for m in methods:
                preds, t_a, t_b = per_method[m]
                times[m].append(t_a + t_b)
                comp_times[m].append(t_b)
                if args.no_accuracy or rep > 0:
                    continue
                m_gt = (gt > 127).astype(np.uint8)
                ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(m_gt, 8)
                gts = []
                for lab in range(1, ncomp):
                    comp = (labels == lab).astype(np.uint8)
                    if comp.sum() < 50:
                        continue
                    gts.append(comp)
                pairs, n_miss, n_fp = match_pairs(preds, gts)
                for pm, gm, iou in pairs:
                    rec = dict(method=m, uid=row.uid, dataset=row.dataset, match_iou=iou,
                               n_gt=len(gts), n_pred=len(preds))
                    rec.update(M.evaluate_pair(pm, gm))
                    records.append(rec)
                if not pairs:
                    records.append(dict(method=m, uid=row.uid, dataset=row.dataset, dice=0.0,
                                        iou=0.0, hd95=np.nan, assd=np.nan, bf1=0.0,
                                        area_err=np.nan, match_iou=0.0, n_gt=len(gts),
                                        n_pred=len(preds)))
                records.append(dict(method=m + "__detstats", uid=row.uid, dataset=row.dataset,
                                    n_gt=len(gts), n_pred=len(preds), n_miss=n_miss, n_fp=n_fp))
        if n % 25 == 0:
            print(f"  {n}/{len(sp)}", flush=True)

    # ---------------- 汇总
    speed = []
    for m in methods:
        t = np.array(times[m]) * 1000.0
        d = np.array(comp_times[m]) * 1000.0
        speed.append(dict(method=m, n=len(t), end2end_ms_mean=t.mean(), end2end_ms_median=np.median(t),
                          end2end_ms_std=t.std(), seg_only_ms_mean=d.mean(), fps=1000.0 / t.mean()))
    spd = pd.DataFrame(speed)
    spd.to_csv(out / "speed.csv", index=False)
    print("\n=== 速度 (每张图端到端, ms) ===")
    print(spd.round(2).to_string(index=False))

    if records:
        df = pd.DataFrame(records)
        df.to_csv(out / "per_image.csv", index=False)
        score = df[~df["method"].str.endswith("__detstats")].copy()
        agg = score.groupby("method").agg(
            n=("dice", "size"), dice=("dice", "mean"), dice_std=("dice", "std"),
            iou=("iou", "mean"), hd95=("hd95", "mean"), assd=("assd", "mean"),
            bf1=("bf1", "mean"), area_err=("area_err", "mean"),
            fail=("dice", lambda s: float((s < 0.5).mean()))).reset_index()
        acc = agg.merge(spd, on="method", how="outer")
        acc.to_csv(out / "accuracy_speed.csv", index=False)
        print("\n=== 精度 + 速度 ===")
        print(acc.round(4).to_string(index=False))
    else:
        acc = spd
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
