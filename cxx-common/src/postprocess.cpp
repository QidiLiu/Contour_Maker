// YOLO 后处理实现。
#include "cxxcommon/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <queue>

#include "cxxcommon/metrics.hpp"

namespace cxxcommon {

std::vector<int> Nms(const std::vector<Box>& boxes, const std::vector<float>& scores,
                     float iou_thresh) {
  std::vector<int> keep;
  const size_t n = boxes.size();
  if (n == 0) return keep;

  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<uint8_t> dead(n, 0);
  for (size_t i = 0; i < n; ++i) {
    if (dead[i]) continue;
    keep.push_back(order[i]);
    const Box& a = boxes[order[i]];
    for (size_t j = i + 1; j < n; ++j) {
      if (dead[j]) continue;
      if (BoxIou(a, boxes[order[j]]) > iou_thresh) dead[j] = 1;
    }
  }
  return keep;
}

namespace {

// letterbox 空间的 cxcywh -> 原图 xyxy
inline Box LetterboxBoxToOriginal(const float* row, float ratio, int pad_x, int pad_y,
                                  int orig_w, int orig_h) {
  const float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
  const float x1 = cx - bw / 2.f, y1 = cy - bh / 2.f;
  const float x2 = cx + bw / 2.f, y2 = cy + bh / 2.f;
  Box b{(x1 - static_cast<float>(pad_x)) / ratio,
       (y1 - static_cast<float>(pad_y)) / ratio,
       (x2 - static_cast<float>(pad_x)) / ratio,
       (y2 - static_cast<float>(pad_y)) / ratio};
  return ClipBox(b, orig_w, orig_h);
}

}  // namespace

std::vector<int> NmsIndices(const std::vector<Box>& boxes, const float* scores,
                            int count, float iou_thresh, float conf_thresh) {
  std::vector<int> order;
  order.reserve(count);
  for (int i = 0; i < count; ++i) {
    if (scores[i] < conf_thresh) continue;
    if (i < static_cast<int>(boxes.size()) && boxes[i].area() <= 0.f) continue;
    order.push_back(i);
  }
  std::sort(order.begin(), order.end(),
            [scores](int a, int b) { return scores[a] > scores[b]; });

  std::vector<uint8_t> dead(order.size(), 0);
  std::vector<int> keep;
  for (size_t i = 0; i < order.size(); ++i) {
    if (dead[i]) continue;
    keep.push_back(order[i]);
    for (size_t j = i + 1; j < order.size(); ++j) {
      if (dead[j]) continue;
      if (BoxIou(boxes[order[i]], boxes[order[j]]) > iou_thresh) dead[j] = 1;
    }
  }
  return keep;
}

void DecodeDetections(const float* boxes, const float* scores, const int* classes,
                      int count, float conf_thresh, int orig_w, int orig_h,
                      float ratio, int pad_x, int pad_y, float nms_iou,
                      YoloDetOut* out) {
  out->boxes.clear();
  out->class_ids.clear();
  out->scores.clear();
  if (count <= 0 || !boxes || !scores || !classes) return;

  // 先算全部框（供 NMS 按 IoU 判定），再按分数降序贪心抑制
  std::vector<Box> all;
  all.reserve(count);
  for (int i = 0; i < count; ++i) {
    all.push_back(LetterboxBoxToOriginal(boxes + static_cast<size_t>(i) * 4,
                                         ratio, pad_x, pad_y, orig_w, orig_h));
  }
  const std::vector<int> keep = NmsIndices(all, scores, count, nms_iou, conf_thresh);
  out->indices = keep;
  for (int k : keep) {
    out->boxes.push_back(all[k]);
    out->class_ids.push_back(classes[k]);
    out->scores.push_back(scores[k]);
  }
}

std::vector<std::vector<uint8_t>> CombineMasks(
    const float* protos, int P, int H, int W,
    const float* coeffs, const std::vector<Box>& boxes_lb,
    int lb_size, int orig_w, int orig_h, float ratio, int pad_x, int pad_y) {
  const size_t n = boxes_lb.size();
  std::vector<std::vector<uint8_t>> masks(n);
  if (n == 0 || P == 0 || H == 0 || W == 0) return masks;

  // prototype 组合 -> 每个实例一张 (H,W) 概率图
  std::vector<float> combined(static_cast<size_t>(H) * W);
  for (size_t i = 0; i < n; ++i) {
    const float* c = coeffs + i * P;
    // 只在框内区域做组合，省去框外的无用计算
    int bx1 = std::max(0, static_cast<int>(std::floor(boxes_lb[i].x1)));
    int by1 = std::max(0, static_cast<int>(std::floor(boxes_lb[i].y1)));
    int bx2 = std::min(W, static_cast<int>(std::ceil(boxes_lb[i].x2)) + 1);
    int by2 = std::min(H, static_cast<int>(std::ceil(boxes_lb[i].y2)) + 1);
    if (bx2 <= bx1 || by2 <= by1) {
      bx1 = std::max(0, static_cast<int>(boxes_lb[i].x1));
      by1 = std::max(0, static_cast<int>(boxes_lb[i].y1));
      bx2 = std::min(W, bx1 + 1);
      by2 = std::min(H, by1 + 1);
    }

    for (int y = by1; y < by2; ++y) {
      for (int x = bx1; x < bx2; ++x) {
        float v = 0.f;
        for (int p = 0; p < P; ++p) {
          v += c[p] * protos[static_cast<size_t>(p) * H * W +
                             static_cast<size_t>(y) * W + x];
        }
        combined[static_cast<size_t>(y) * W + x] = v;
      }
    }

    // -> 原图分辨率掩码（只在框对应的区域内上采样）
    std::vector<uint8_t>& m = masks[i];
    m.assign(static_cast<size_t>(orig_w) * orig_h, 0);
    // 掩码坐标系：prototype 网格 -> letterbox 画布 -> 原图
    const float proto_to_lb_x = static_cast<float>(lb_size) / W;
    const float proto_to_lb_y = static_cast<float>(lb_size) / H;
    const int ox1 = std::max(0, static_cast<int>(std::floor(
                                     (boxes_lb[i].x1) * proto_to_lb_x)));
    const int oy1 = std::max(0, static_cast<int>(std::floor(
                                     (boxes_lb[i].y1) * proto_to_lb_y)));
    const int ox2 = std::min(lb_size, static_cast<int>(std::ceil(
                                           boxes_lb[i].x2 * proto_to_lb_x)));
    const int oy2 = std::min(lb_size, static_cast<int>(std::ceil(
                                           boxes_lb[i].y2 * proto_to_lb_y)));

    for (int oy = oy1; oy < oy2; ++oy) {
      const int py = std::min(H - 1, static_cast<int>(oy / proto_to_lb_y));
      // letterbox -> 原图
      const int gy = static_cast<int>(std::lround((oy - pad_y) / ratio));
      if (gy < 0 || gy >= orig_h) continue;
      for (int ox = ox1; ox < ox2; ++ox) {
        const int px = std::min(W - 1, static_cast<int>(ox / proto_to_lb_x));
        const int gx = static_cast<int>(std::lround((ox - pad_x) / ratio));
        if (gx < 0 || gx >= orig_w) continue;
        if (combined[static_cast<size_t>(py) * W + px] > 0.f) {
          m[static_cast<size_t>(gy) * orig_w + gx] = 1;
        }
      }
    }
  }
  return masks;
}

void FilterMasks(std::vector<std::vector<uint8_t>>* masks, int w, int h, int min_area) {
  for (auto& m : *masks) {
    auto comps = MaskToComponents(m, w, h, min_area);
    std::fill(m.begin(), m.end(), uint8_t{0});
    if (!comps.empty()) {
      // 取最大连通域
      size_t best = 0;
      for (size_t i = 1; i < comps.size(); ++i) {
        if (comps[i].area > comps[best].area) best = i;
      }
      m = comps[best].mask;
    }
  }
}

}  // namespace cxxcommon