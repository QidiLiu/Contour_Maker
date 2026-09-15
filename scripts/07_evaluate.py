"""统一评估: 对同一套 test split, 在完全相同的指标下对比

  * rough_otsu          : YOLO26n 检测框 + Otsu 自适应分割 (方案 A 的粗糙轮廓, 无精细化)
  * refiner             : YOLO26n 检测框 + Otsu 粗糙轮廓 + contour-refiner (方案 A 完整)
  * refiner_gtbox       : GT 框 + Otsu + contour-refiner (分离检测误差的参照)
  * yolo26n_seg         : YOLO26n-seg 端到端实例分割 (方案 B)

用法示例:
    PYTHONPATH=src ./.conda/bin/python scripts/07_evaluate.py \
        --refiner runs/refiner_A5/best.pt --det runs/det_yolo26n/weights/best.pt \
        --seg runs/seg_yolo26n/weights/best.pt --out reports/eval_A5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cm import geometry as G  # noqa: E402
from cm import metrics as M  # noqa: E402
from cm import roi as R  # noqa: E402
from cm.config import IOU_MATCH_THRESHOLD, REPORTS, RUNS, SPLITS, UNIFIED, AugConfig, RefinerConfig  # noqa: E402
from cm.model import build_refiner  # noqa: E402


# ---------------------------------------------------------------- model load
def load_refiner(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    rcfg = RefinerConfig(**ckpt["rcfg"])
    model = build_refiner(rcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, rcfg, ckpt


@torch.no_grad()
def refine_contour(model, rcfg: RefinerConfig, image: np.ndarray, bbox, rough_pts: np.ndarray,
                   device) -> np.ndarray:
    """用 contour-refiner 精细化粗糙轮廓, 返回全图坐标的精准轮廓点。"""
    H, W = image.shape[:2]
    diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    window = max(6.0, rcfg.window_ratio * diag)
    patches, frames = R.extract_patches(image, rough_pts, window, rcfg.roi_size, normalize=rcfg.zscore)
    rough_norm = R.norm_from_frame(rough_pts, frames)
    aux = R.aux_features(rough_pts, bbox, (H, W))
    centers = np.stack([f.center_norm for f in frames])
    t = lambda a: torch.from_numpy(a[None]).float().to(device)  # noqa: E731
    pred = model(t(patches), t(rough_norm), t(aux), t(centers))[0].float().cpu().numpy()
    return R.frame_from_norm(pred, frames)


def gt_targets(mask: np.ndarray, min_area: int = 50):
    m = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        bb = M.mask_bbox(comp)
        contour = G.mask_to_contour(comp)
        if bb is None or contour is None:
            continue
        out.append(dict(mask=comp, bbox=bb, contour=G.ensure_ccw(G.resample_closed(contour, 256))))
    return out


# ---------------------------------------------------------------- box sources
def boxes_from_yolo(model, image_path: Path, conf: float, device, imgsz: int):
    res = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False, device=device)
    r = res[0]
    boxes = []
    masks = []
    if r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()
        for i, b in enumerate(xyxy):
            boxes.append(tuple(float(v) for v in b))
        if getattr(r, "masks", None) is not None and r.masks is not None:
            for poly in r.masks.xy:
                masks.append(np.asarray(poly, np.float32))
    return boxes, masks


def _mean_pt_dist(pts: np.ndarray, ref_contour: np.ndarray) -> float:
    """轮廓点到参考轮廓(密集点集)的平均最近距离 (像素)。"""
    if pts is None or len(pts) == 0 or ref_contour is None or len(ref_contour) == 0:
        return float("nan")
    p = np.asarray(pts, np.float32)
    r = np.asarray(ref_contour, np.float32)
    d = np.sqrt(((p[:, None, :] - r[None, :, :]) ** 2).sum(-1)).min(1)
    return float(d.mean())


def match_and_score(preds: list[dict], gts: list[dict]) -> tuple[list[dict], int, int]:
    """按 mask IoU 贪心匹配预测与 GT, 返回已匹配的 (pred, gt) 对与漏检/误检数。"""
    if not preds or not gts:
        return [], len(gts), len(preds)
    iou_mat = np.zeros((len(preds), len(gts)), np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = M.iou(p["mask"], g["mask"])
    pairs = []
    used_p, used_g = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou_mat, axis=None), iou_mat.shape))[0]
    for i, j in order:
        if i in used_p or j in used_g:
            continue
        if iou_mat[i, j] < IOU_MATCH_THRESHOLD:
            continue
        used_p.add(int(i)); used_g.add(int(j))
        pairs.append(dict(pred=preds[int(i)], gt=gts[int(j)], match_iou=float(iou_mat[i, j])))
    return pairs, len(gts) - len(used_g), len(preds) - len(used_p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiner", type=str, default=None)
    ap.add_argument("--det", type=str, default=None, help="YOLO26n 检测权重")
    ap.add_argument("--seg", type=str, default=None, help="YOLO26n-seg 权重")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(REPORTS / "eval"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--uids-file", default=None,
                    help="限定评估图片的 uid 清单 csv (例如 data/yolo_subset/_subset_splits.csv), "
                         "保证与 YOLO 训练子集一致")
    ap.add_argument("--save-vis", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    sp = pd.read_csv(SPLITS / "splits.csv")
    sp = sp[sp["split"] == args.split]
    if args.datasets:
        sp = sp[sp["dataset"].isin(args.datasets)]
    if args.uids_file:
        keep = set(pd.read_csv(args.uids_file)["uid"].astype(str))
        before = len(sp)
        sp = sp[sp["uid"].astype(str).isin(keep)]
        print(f"[eval] uid 过滤: {before} -> {len(sp)}")
    if args.max_images:
        sp = sp.head(args.max_images)
    print(f"[eval] split={args.split} images={len(sp)} device={device}", flush=True)

    refiner = rcfg = None
    if args.refiner:
        refiner, rcfg, ckpt = load_refiner(Path(args.refiner), device)
        print(f"[eval] refiner={args.refiner} level={ckpt.get('level')} val_dice={ckpt.get('val_dice')}",
              flush=True)

    det_model = seg_model = None
    if args.det:
        from ultralytics import YOLO
        det_model = YOLO(args.det)
    if args.seg:
        from ultralytics import YOLO
        seg_model = YOLO(args.seg)

    records: list[dict] = []
    vis_saved = 0
    for n_done, (_, row) in enumerate(sp.iterrows(), start=1):
        img_path = UNIFIED / row["image"]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(UNIFIED / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or gt_mask is None:
            continue
        gts = gt_targets(gt_mask)
        if not gts:
            continue
        H, W = image.shape[:2]

        # ---------------- 检测框来源 (方案 A 用检测模型)
        det_boxes: list = []
        if det_model is not None:
            det_boxes, _ = boxes_from_yolo(det_model, img_path, args.conf, str(device), args.imgsz)
        elif seg_model is not None:
            det_boxes, _ = boxes_from_yolo(seg_model, img_path, args.conf, str(device), args.imgsz)

        variants: dict[str, list[dict]] = {}

        # 方案 B: YOLO26n-seg (独立推理, 取其自身 mask 多边形)
        if seg_model is not None:
            _, seg_polys = boxes_from_yolo(seg_model, img_path, args.conf, str(device), args.imgsz)
            preds = []
            for poly in seg_polys:
                if poly is None or len(poly) < 3:
                    continue
                mk = G.contour_to_mask(poly, (H, W))
                if mk.sum() == 0:
                    continue
                preds.append(dict(mask=mk, contour=poly, bbox=M.mask_bbox(mk)))
            variants["yolo26n_seg"] = preds

        # 方案 A: Otsu 粗糙轮廓 (+ refiner)
        def run_pipeline(box_list, tag: str, use_refiner: bool):
            preds = []
            n_pts = rcfg.n_points if (use_refiner and rcfg is not None) else 256
            for b in box_list:
                rr = R.otsu_adaptive_rough_contour(image, b, polarity="dark")
                if not rr.ok or len(rr.contour) < 3:
                    continue
                rough_pts = G.resample_closed(G.ensure_ccw(rr.contour), n_pts)
                if use_refiner and refiner is not None:
                    pts = refine_contour(refiner, rcfg, image, b, rough_pts, device)
                else:
                    pts = rough_pts
                mk = G.contour_to_mask(pts, (H, W))
                if mk.sum() == 0:
                    continue
                preds.append(dict(mask=mk, contour=pts, bbox=M.mask_bbox(mk), rough=rr,
                                  rough_pts=rough_pts))
            return preds

        if det_model is not None and refiner is not None:
            variants["refiner"] = run_pipeline(det_boxes, "refiner", True)
            variants["rough_otsu"] = run_pipeline(det_boxes, "rough", False)
        # 纯 Otsu 对照 (无 refiner); 框来源优先用 YOLO 检测框, 便于与方案 B 同条件比较
        if det_model is None or refiner is None:
            base_boxes = det_boxes if det_boxes else ([g["bbox"] for g in gts] if refiner is None else [])
            if base_boxes:
                variants["rough_otsu"] = run_pipeline(base_boxes, "rough", False)
        if refiner is not None:
            gt_boxes = [g["bbox"] for g in gts]
            variants["refiner_gtbox"] = run_pipeline(gt_boxes, "refiner_gt", True)
            variants["rough_otsu_gtbox"] = run_pipeline(gt_boxes, "rough_gt", False)

        # ---------------- 打分
        for name, preds in variants.items():
            pairs, n_miss, n_fp = match_and_score(preds, gts)
            for pr in pairs:
                rec = dict(method=name, uid=row["uid"], dataset=row["dataset"], modality=row["modality"],
                           match_iou=pr["match_iou"], n_gt=len(gts), n_pred=len(preds))
                rec.update(M.evaluate_pair(pr["pred"]["mask"], pr["gt"]["mask"]))
                # 轮廓点级诊断: (预测点/粗糙点) 到 GT 轮廓的平均像素距离
                gt_c = pr["gt"]["contour"]
                for tag, key in (("pred", "contour"), ("rough", "rough_pts")):
                    pts = pr["pred"].get(key)
                    if pts is None or len(pts) == 0:
                        continue
                    d = _mean_pt_dist(pts, gt_c)
                    rec[f"{tag}_pt_err"] = d
                records.append(rec)
            if not pairs:
                records.append(dict(method=name, uid=row["uid"], dataset=row["dataset"],
                                    modality=row["modality"], dice=0.0, iou=0.0, hd95=np.nan,
                                    assd=np.nan, bf1=0.0, area_err=np.nan, match_iou=0.0,
                                    n_gt=len(gts), n_pred=len(preds)))
            # 记录检测层面的漏检/误检
            records.append(dict(method=name + "__detstats", uid=row["uid"], dataset=row["dataset"],
                                modality=row["modality"], n_gt=len(gts), n_pred=len(preds),
                                n_miss=n_miss, n_fp=n_fp, dice=np.nan))

        # ---------------- 可视化
        if vis_saved < args.save_vis and "refiner" in variants:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for g in gts:
                cv2.polylines(vis, [np.round(g["contour"]).astype(np.int32)], True, (0, 255, 0), 1)
            for p in variants.get("rough_otsu", []):
                cv2.polylines(vis, [np.round(p["contour"]).astype(np.int32)], True, (0, 165, 255), 1)
            for p in variants.get("refiner", []):
                cv2.polylines(vis, [np.round(p["contour"]).astype(np.int32)], True, (255, 0, 0), 1)
            for p in variants.get("yolo26n_seg", []):
                cv2.polylines(vis, [np.round(p["contour"]).astype(np.int32)], True, (255, 0, 255), 1)
            cv2.imwrite(str(out_dir / f"vis_{row['dataset']}_{row['uid']}.png"), vis)
            vis_saved += 1
        if n_done % 50 == 0:
            print(f"  ... {n_done}/{len(sp)}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "per_image.csv", index=False)
    if df.empty or "method" not in df.columns:
        print("[eval] 没有任何可用方法 (既无 refiner 也无 seg 权重), 请检查参数")
        return 1
    score = df[~df["method"].str.endswith("__detstats")].copy()
    summary = score.groupby(["method", "dataset"]).agg(
        n=("dice", "size"), dice=("dice", "mean"), iou=("iou", "mean"),
        hd95=("hd95", "mean"), assd=("assd", "mean"), bf1=("bf1", "mean"),
        area_err=("area_err", "mean")).reset_index()
    overall = score.groupby(["method"]).agg(
        n=("dice", "size"), dice=("dice", "mean"), dice_std=("dice", "std"), iou=("iou", "mean"),
        hd95=("hd95", "mean"), assd=("assd", "mean"), bf1=("bf1", "mean"),
        area_err=("area_err", "mean")).reset_index()
    det = df[df["method"].str.endswith("__detstats")]
    det_sum = det.groupby("method").agg(n_gt=("n_gt", "sum"), n_pred=("n_pred", "sum"),
                                        n_miss=("n_miss", "sum"), n_fp=("n_fp", "sum")).reset_index()
    for d in (summary, overall, det_sum):
        d["method"] = d["method"].str.replace("__detstats", "", regex=False)
    summary.to_csv(out_dir / "summary_by_dataset.csv", index=False)
    overall.to_csv(out_dir / "summary_overall.csv", index=False)
    det_sum.to_csv(out_dir / "detection_stats.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print("\n=== overall ===")
    print(overall.to_string(index=False))
    print("\n=== by dataset ===")
    print(summary.to_string(index=False))
    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
