// 两阶段流水线实现。
#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cxxcommon/eval_harness.hpp"
#include "cxxcommon/metrics.hpp"
#include "cxxcommon/postprocess.hpp"

namespace fs = std::filesystem;
using namespace cxxcommon;

namespace detunet {
namespace {

// YOLO26-det 的 ONNX 导出：
//   output0: (1, N, 6)  每行 = [x1, y1, x2, y2, score, class_id]
//            N = max_det(300)，已按 score 降序（头部自带 TopK），未做 NMS
//            坐标为 letterbox 空间 xyxy
struct DetHead {
  const float* rows = nullptr;   // (N, cols)
  int N = 0;
  int cols = 0;
};

bool ParseDetHead(const std::vector<int>& shape, const float* data, int num_classes,
                  DetHead* o, std::string* err) {
  if (!data || shape.empty()) {
    *err = "检测模型无输出";
    return false;
  }
  if (shape.size() != 3) {
    *err = "检测输出应为 3 维 (1,N,C)，实际维度 " + std::to_string(shape.size());
    return false;
  }
  o->N = shape[1];
  o->cols = shape[2];
  o->rows = data;
  if (o->cols < 6) {
    *err = "检测输出列数 " + std::to_string(o->cols) + " < 6 (box4+score+class)";
    return false;
  }
  (void)num_classes;
  return true;
}

}  // namespace

namespace detail {

struct StageTiming {
  double det_ms = 0, roi_ms = 0, seg_ms = 0, post_ms = 0;
};

std::vector<Instance> InferTwoStage(MnnRunner* det, MnnRunner* mk, const Config& cfg,
                                    const Image& img, StageTiming* t,
                                    std::string* err) {
  std::vector<Instance> out;

  // ================= 第一级：YOLO26n 检测 =================
  LetterboxInfo lb;
  const Image canvas = Letterbox(img, cfg.det_imgsz, &lb);
  // 检测模型输入为 3 通道（ultralytics 以 BGR 读图）。灰度复制到 3 通道即可。
  const size_t det_plane = static_cast<size_t>(cfg.det_imgsz) * cfg.det_imgsz;
  std::vector<float> din(det_plane * 3);
  for (int y = 0; y < cfg.det_imgsz; ++y) {
    for (int x = 0; x < cfg.det_imgsz; ++x) {
      const float v = static_cast<float>(canvas.at(y, x)[0]) / 255.f;
      const size_t i = static_cast<size_t>(y) * cfg.det_imgsz + x;
      din[i] = v;
      din[det_plane + i] = v;
      din[2 * det_plane + i] = v;
    }
  }
  if (!det->Forward(din.data(), {1, 3, cfg.det_imgsz, cfg.det_imgsz}, err)) {
    return out;
  }
  if (t) t->det_ms += det->LastLatencyMs();

  const float* det_data = nullptr;
  std::vector<int> det_shape;
  if (!det->Output(0, &det_data, &det_shape)) {
    *err = "取检测输出失败";
    return out;
  }
  DetHead head;
  if (!ParseDetHead(det_shape, det_data, cfg.num_classes, &head, err)) return out;
  if (head.N == 0) return out;

  // ---- 逐行解析: [x1,y1,x2,y2, score, class_id]，反 letterbox 到原图
  YoloDetOut boxes;
  {
    std::vector<Box> all;
    std::vector<int> cls_all;
    std::vector<float> score_all;
    all.reserve(head.N);
    for (int i = 0; i < head.N; ++i) {
      const float* r = head.rows + static_cast<size_t>(i) * head.cols;
      const float score = r[4];
      if (score < cfg.conf) continue;
      const int cls = static_cast<int>(std::lround(r[5]));
      if (cls < 0 || cls >= cfg.num_classes) continue;
      Box lbx{r[0], r[1], r[2], r[3]};
      if (lbx.area() <= 0.f) continue;
      Box ob{(lbx.x1 - lb.pad_x) / lb.ratio, (lbx.y1 - lb.pad_y) / lb.ratio,
             (lbx.x2 - lb.pad_x) / lb.ratio, (lbx.y2 - lb.pad_y) / lb.ratio};
      ob = ClipBox(ob, img.width, img.height);
      if (ob.area() <= 0.f) continue;
      all.push_back(ob);
      cls_all.push_back(cls);
      score_all.push_back(score);
    }
    if (all.empty()) return out;
    // 按分数降序 + 类内 NMS
    std::vector<int> order(all.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return score_all[a] > score_all[b]; });
    std::vector<uint8_t> dead(order.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
      if (dead[i]) continue;
      for (size_t j = i + 1; j < order.size(); ++j) {
        if (dead[j]) continue;
        if (cls_all[order[i]] != cls_all[order[j]]) continue;   // 类内 NMS
        if (BoxIou(all[order[i]], all[order[j]]) > cfg.nms_iou) dead[j] = 1;
      }
    }
    for (size_t i = 0; i < order.size(); ++i) {
      if (dead[i]) continue;
      boxes.boxes.push_back(all[order[i]]);
      boxes.class_ids.push_back(cls_all[order[i]]);
      boxes.scores.push_back(score_all[order[i]]);
    }
  }
  if (boxes.boxes.empty()) return out;

  // ================= 第二级：ROI 裁剪 + MK-UNet =================
  const auto t_roi0 = std::chrono::steady_clock::now();
  std::vector<Image> rois;
  std::vector<Box> roi_boxes;
  std::vector<int> roi_classes;
  std::vector<float> roi_scores;
  for (size_t bi = 0; bi < boxes.boxes.size(); ++bi) {
    const Box& b = boxes.boxes[bi];
    const float bw = b.x2 - b.x1, bh = b.y2 - b.y1;
    // 向外扩 pad_ratio，给边界留上下文（与 Python 侧一致）
    Box eb{b.x1 - bw * cfg.pad_ratio, b.y1 - bh * cfg.pad_ratio,
           b.x2 + bw * cfg.pad_ratio, b.y2 + bh * cfg.pad_ratio};
    eb = ClipBox(eb, img.width, img.height);
    Image roi;
    if (!CropRoi(img, eb, &roi)) continue;
    rois.push_back(ResizeRoi(roi, cfg.roi_size, /*square_pad=*/true));
    roi_boxes.push_back(eb);
    roi_classes.push_back(boxes.class_ids[bi]);
    roi_scores.push_back(boxes.scores[bi]);
  }
  if (t) {
    t->roi_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_roi0).count();
  }
  if (rois.empty()) return out;

  // MK-UNet 前向（逐 ROI，batch=1，与 Python 侧逐个 refine 的结构一致）
  std::vector<std::vector<uint8_t>> full_masks;
  full_masks.reserve(rois.size());
  std::vector<float> min(static_cast<size_t>(cfg.roi_size) * cfg.roi_size);
  const auto t_seg0 = std::chrono::steady_clock::now();
  for (const auto& roi : rois) {
    for (int y = 0; y < cfg.roi_size; ++y) {
      for (int x = 0; x < cfg.roi_size; ++x) {
        min[static_cast<size_t>(y) * cfg.roi_size + x] =
            static_cast<float>(roi.at(y, x)[0]) / 255.f;
      }
    }
    if (!mk->Forward(min.data(), {1, 1, cfg.roi_size, cfg.roi_size}, err)) {
      return out;
    }
    const float* logit = nullptr;
    std::vector<int> mshape;
    if (!mk->Output(0, &logit, &mshape) || !logit) continue;

    // sigmoid(logit) > thr -> 二值 ROI 掩码
    const size_t n = static_cast<size_t>(cfg.roi_size) * cfg.roi_size;
    std::vector<uint8_t> roi_mask(n, 0);
    for (size_t i = 0; i < n; ++i) {
      roi_mask[i] = (1.f / (1.f + std::exp(-logit[i])) > cfg.thr) ? 1 : 0;
    }

    std::vector<uint8_t> full;
    PasteRoiMask(roi_mask, cfg.roi_size, roi_boxes[full_masks.size()],
                 img.width, img.height, &full);
    full_masks.push_back(std::move(full));
  }
  if (t) {
    t->seg_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t_seg0).count();
  }

  // ================= 后处理：过滤 + 轮廓 =================
  const auto t_post0 = std::chrono::steady_clock::now();
  std::vector<std::vector<uint8_t>> masks = full_masks;
  FilterMasks(&masks, img.width, img.height, cfg.mask_min_area);
  for (size_t i = 0; i < masks.size() && i < roi_boxes.size(); ++i) {
    Instance inst;
    inst.class_id = roi_classes[i];
    inst.score = roi_scores[i];
    inst.box = roi_boxes[i];
    inst.mask = masks[i];
    inst.mask_w = img.width;
    inst.mask_h = img.height;
    inst.has_mask = true;
    if (MaskToContour(inst.mask, img.width, img.height, &inst.contour)) {
      out.push_back(std::move(inst));
    }
  }
  if (t) {
    t->post_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t_post0).count();
  }
  return out;
}

}  // namespace detail

bool RunOne(const Config& cfg, const std::string& image_path, std::string* err) {
  MnnRunner det, mk;
  Backend a1 = Backend::kCpu, a2 = Backend::kCpu;
  if (!det.Load(cfg.det_model, Backend::kCpu, cfg.threads, &a1, err)) return false;
  if (!mk.Load(cfg.mkunet_model, Backend::kCpu, cfg.threads, &a2, err)) return false;
  Image img;
  if (!LoadImage(image_path, &img, err)) return false;
  detail::StageTiming t;
  auto insts = detail::InferTwoStage(&det, &mk, cfg, img, &t, err);
  if (!err->empty()) return false;
  printf("%s -> %zu instances (det %.2f + roi %.2f + seg %.2f + post %.2f ms)\n",
         image_path.c_str(), insts.size(), t.det_ms, t.roi_ms, t.seg_ms, t.post_ms);
  for (const auto& in : insts) {
    printf("  class=%d score=%.3f box=[%.1f %.1f %.1f %.1f] contour_pts=%zu\n",
           in.class_id, in.score, in.box.x1, in.box.y1, in.box.x2, in.box.y2,
           in.contour.size());
  }
  return true;
}

Summary RunPipeline(const Config& cfg, bool verbose) {
  Summary sum;
  std::string err;

  MnnRunner det, mk;
  Backend a1 = Backend::kCpu, a2 = Backend::kCpu;
  if (!det.Load(cfg.det_model, Backend::kCpu, cfg.threads, &a1, &err)) {
    fprintf(stderr, "[error] detector: %s\n", err.c_str());
    return sum;
  }
  if (!mk.Load(cfg.mkunet_model, Backend::kCpu, cfg.threads, &a2, &err)) {
    fprintf(stderr, "[error] MK-UNet: %s\n", err.c_str());
    return sum;
  }
  printf("[model] det   = %s (backend=%s)\n", cfg.det_model.c_str(), BackendName(a1));
  printf("[model] mkunet= %s (backend=%s) roi=%d\n", cfg.mkunet_model.c_str(),
         BackendName(a2), cfg.roi_size);

  // Warm up both models before measuring.
  {
    std::vector<float> w1(static_cast<size_t>(cfg.det_imgsz) * cfg.det_imgsz * 3, 0.5f);
    std::vector<float> w2(static_cast<size_t>(cfg.roi_size) * cfg.roi_size, 0.5f);
    for (int i = 0; i < cfg.warmup; ++i) {
      if (!det.Forward(w1.data(), {1, 3, cfg.det_imgsz, cfg.det_imgsz}, &err)) {
        fprintf(stderr, "[error] detector warmup: %s\n", err.c_str());
        return sum;
      }
      if (!mk.Forward(w2.data(), {1, 1, cfg.roi_size, cfg.roi_size}, &err)) {
        fprintf(stderr, "[error] MK-UNet warmup: %s\n", err.c_str());
        return sum;
      }
    }
  }

  EvalConfig ec;
  ec.image_dir = cfg.image_dir;
  ec.list_file = cfg.list_file;
  ec.gt_dir = cfg.gt_dir;
  ec.gt_instances_dir = cfg.gt_instances_dir;
  ec.out_dir = cfg.out_dir;
  ec.max_images = cfg.max_images;
  ec.min_area = cfg.mask_min_area;
  ec.verbose = verbose;

  double det_ms = 0, roi_ms = 0, seg_ms = 0;
  const auto infer = [&](const Image& img, std::vector<Instance>* out,
                         double* infer_ms, std::map<std::string, double>* stages,
                         std::string* e) -> bool {
    detail::StageTiming t;
    auto insts = detail::InferTwoStage(&det, &mk, cfg, img, &t, e);
    if (!e->empty()) return false;
    det_ms += t.det_ms;
    roi_ms += t.roi_ms;
    seg_ms += t.seg_ms;
    *infer_ms = t.det_ms + t.seg_ms;
    if (stages) {
      (*stages)["detection"] += t.det_ms;
      (*stages)["roi"] += t.roi_ms;
      (*stages)["mkunet"] += t.seg_ms;
      (*stages)["post"] += t.post_ms;
    }
    *out = std::move(insts);
    return true;
  };

  const EvalSummary es = RunEvaluation(ec, infer);
  sum.n_images = es.n_images;
  sum.n_gt = es.n_gt;
  sum.n_pred = es.n_pred;
  sum.n_tp = es.n_tp;
  sum.n_fp = es.n_fp;
  sum.n_fn = es.n_fn;
  sum.n_cls_ok = es.n_cls_ok;
  sum.dice = es.dice;
  sum.iou = es.iou;
  sum.hd95 = es.hd95;
  sum.assd = es.assd;
  sum.bf1 = es.bf1;
  sum.area_err = es.area_err;
  sum.n_matched = es.n_matched;
  sum.det_ms = det_ms;
  sum.roi_ms = roi_ms;
  sum.seg_ms = seg_ms;
  sum.post_ms = es.post_ms;
  sum.total_ms = es.total_ms;
  sum.stage_ms = es.stage_ms;
  return sum;
}

}  // namespace detunet
