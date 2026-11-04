// RF-DETR-Seg-Nano pipeline implementation (MNN).
//
// ONNX/MNN interface (exported by rfdetr-seg-train/export_onnx.py):
//   input   (1, 3, R, R)        RGB, mean/std normalised, R = resolution
//   dets    (1, Q, 4)           normalised cx, cy, w, h
//   labels  (1, Q, C + 1)       class logits (sigmoid -> score, last = background)
//   masks   (1, Q, R/4, R/4)    per-query mask logits
//
// RF-DETR is a DETR-style detector: it does not need NMS.  The reference Python
// post-processing thresholds sigmoid(labels) and bilinearly upsamples each mask
// logit map to the original image size before thresholding at 0.
#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "cxxcommon/eval_harness.hpp"
#include "cxxcommon/metrics.hpp"

using namespace cxxcommon;

namespace rfdetrseg {
namespace {

struct RfHead {
  const float* dets = nullptr;
  int Q = 0;
  const float* labels = nullptr;
  int ncols = 0;
  const float* masks = nullptr;
  int Hm = 0;
  int Wm = 0;
};

bool ParseHead(const std::vector<std::vector<int>>& shapes,
               const std::vector<const float*>& data, RfHead* o, std::string* err) {
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& s = shapes[i];
    if (s.size() == 3 && s[2] == 4 && !o->dets) {
      o->dets = data[i];
      o->Q = s[1];
    } else if (s.size() == 3 && !o->labels) {
      o->labels = data[i];
      o->ncols = s[2];
      if (o->Q == 0) o->Q = s[1];
    } else if (s.size() == 4 && !o->masks) {
      o->masks = data[i];
      o->Hm = s[2];
      o->Wm = s[3];
      if (o->Q == 0) o->Q = s[1];
    }
  }
  if (!o->dets || !o->labels || !o->masks || o->Q <= 0 || o->Hm <= 0) {
    *err = "cannot identify RF-DETR outputs (dets/labels/masks)";
    return false;
  }
  return true;
}

// Bilinear resize of a float plane, align_corners=False (matches PyTorch).
void ResizeFloatBilinear(const float* src, int sw, int sh, int dw, int dh,
                         std::vector<float>* dst) {
  dst->assign(static_cast<size_t>(dw) * dh, 0.f);
  if (sw <= 0 || sh <= 0) return;
  const float sx = static_cast<float>(sw) / dw;
  const float sy = static_cast<float>(sh) / dh;
  for (int y = 0; y < dh; ++y) {
    const float fy = (y + 0.5f) * sy - 0.5f;
    const int y0 = std::max(0, std::min(sh - 1, static_cast<int>(std::floor(fy))));
    const int y1 = std::max(0, std::min(sh - 1, y0 + 1));
    const float wy = fy - std::floor(fy);
    for (int x = 0; x < dw; ++x) {
      const float fx = (x + 0.5f) * sx - 0.5f;
      const int x0 = std::max(0, std::min(sw - 1, static_cast<int>(std::floor(fx))));
      const int x1 = std::max(0, std::min(sw - 1, x0 + 1));
      const float wx = fx - std::floor(fx);
      const float v00 = src[static_cast<size_t>(y0) * sw + x0];
      const float v10 = src[static_cast<size_t>(y0) * sw + x1];
      const float v01 = src[static_cast<size_t>(y1) * sw + x0];
      const float v11 = src[static_cast<size_t>(y1) * sw + x1];
      const float top = v00 + (v10 - v00) * wx;
      const float bot = v01 + (v11 - v01) * wx;
      (*dst)[static_cast<size_t>(y) * dw + x] = top + (bot - top) * wy;
    }
  }
}

std::vector<Instance> Infer(MnnRunner* runner, const Config& cfg, const Image& img,
                            double* infer_ms, std::string* err) {
  std::vector<Instance> out;
  const int R = cfg.resolution;
  // RF-DETR preprocessing is a plain square resize (no letterbox/padding).
  // Resize in float to match torchvision's F.resize(antialias=False) rather
  // than quantising to uint8 first.
  static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};  // RGB order
  static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
  const size_t plane = static_cast<size_t>(R) * R;
  std::vector<float> src(img.data.begin(), img.data.end());
  std::vector<float> resized;
  ResizeFloatBilinear(src.data(), img.width, img.height, R, R, &resized);
  std::vector<float> input(plane * 3);
  for (size_t i = 0; i < plane; ++i) {
    const float g = resized[i] / 255.f;
    for (int c = 0; c < 3; ++c) input[c * plane + i] = (g - kMean[c]) / kStd[c];
  }
  if (!runner->Forward(input.data(), {1, 3, R, R}, err)) return out;
  if (infer_ms) *infer_ms = runner->LastLatencyMs();

  std::vector<std::vector<int>> shapes(runner->NumOutputs());
  std::vector<const float*> datas(runner->NumOutputs(), nullptr);
  for (int i = 0; i < runner->NumOutputs(); ++i) {
    if (!runner->Output(i, &datas[i], &shapes[i])) {
      *err = "failed to read output " + runner->OutputName(i);
      return out;
    }
  }
  RfHead head;
  if (!ParseHead(shapes, datas, &head, err)) return out;

  const int W = img.width, H = img.height;
  const int ncls = std::min(cfg.num_classes, std::max(1, head.ncols - 1));
  std::vector<float> mask_plane;
  std::vector<uint8_t> full(static_cast<size_t>(W) * H, 0);

  for (int q = 0; q < head.Q; ++q) {
    const float* logits = head.labels + static_cast<size_t>(q) * head.ncols;
    // sigmoid over the real classes; the last column is background.
    int best_c = -1;
    float best_s = -1.f;
    for (int c = 0; c < ncls; ++c) {
      const float s = 1.f / (1.f + std::exp(-logits[c]));
      if (s > best_s) {
        best_s = s;
        best_c = c;
      }
    }
    if (best_c < 0 || best_s < cfg.conf) continue;

    const float* b = head.dets + static_cast<size_t>(q) * 4;
    const float cx = b[0] * W, cy = b[1] * H;
    const float bw = b[2] * W, bh = b[3] * H;
    Instance inst;
    inst.class_id = best_c;
    inst.score = best_s;
    inst.box = ClipBox(Box{cx - bw / 2.f, cy - bh / 2.f, cx + bw / 2.f, cy + bh / 2.f},
                       W, H);
    if (inst.box.area() <= 0.f) continue;

    const float* mptr = head.masks + static_cast<size_t>(q) * head.Hm * head.Wm;
    ResizeFloatBilinear(mptr, head.Wm, head.Hm, W, H, &mask_plane);
    std::fill(full.begin(), full.end(), 0);
    size_t area = 0;
    for (size_t i = 0; i < full.size(); ++i) {
      if (mask_plane[i] > 0.f) {
        full[i] = 1;
        ++area;
      }
    }
    if (area == 0) continue;
    inst.mask = full;
    inst.mask_w = W;
    inst.mask_h = H;
    inst.has_mask = true;
    if (MaskToContour(inst.mask, W, H, &inst.contour)) out.push_back(std::move(inst));
  }
  return out;
}

}  // namespace

bool RunOne(const Config& cfg, const std::string& image_path, std::string* err) {
  MnnRunner runner;
  Backend actual = Backend::kCpu;
  if (!runner.Load(cfg.model_path, Backend::kCpu, cfg.threads, &actual, err)) return false;
  Image img;
  if (!LoadImage(image_path, &img, err)) return false;
  double ms = 0;
  auto insts = Infer(&runner, cfg, img, &ms, err);
  if (!err->empty()) return false;
  printf("%s -> %zu instances, %.2f ms\n", image_path.c_str(), insts.size(), ms);
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

  MnnRunner runner;
  Backend actual = Backend::kCpu;
  if (!runner.Load(cfg.model_path, Backend::kCpu, cfg.threads, &actual, &err)) {
    fprintf(stderr, "[error] %s\n", err.c_str());
    return sum;
  }
  printf("[model] %s  backend=%s threads=%d resolution=%d conf=%.2f\n",
         cfg.model_path.c_str(), BackendName(actual), cfg.threads, cfg.resolution,
         cfg.conf);

  {
    std::vector<float> warm(static_cast<size_t>(cfg.resolution) * cfg.resolution * 3, 0.f);
    for (int i = 0; i < cfg.warmup; ++i) {
      if (!runner.Forward(warm.data(), {1, 3, cfg.resolution, cfg.resolution}, &err)) {
        fprintf(stderr, "[error] warmup failed: %s\n", err.c_str());
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
  ec.verbose = verbose;

  const auto infer = [&](const Image& img, std::vector<Instance>* out,
                         double* infer_ms, std::map<std::string, double>* stages,
                         std::string* e) -> bool {
    auto insts = Infer(&runner, cfg, img, infer_ms, e);
    if (!e->empty()) return false;
    if (stages) (*stages)["model"] += *infer_ms;
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
  sum.infer_ms = es.infer_ms;
  sum.post_ms = es.post_ms;
  sum.total_ms = es.total_ms;
  sum.stage_ms = es.stage_ms;
  return sum;
}

}  // namespace rfdetrseg
