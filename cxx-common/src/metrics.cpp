// C++ 侧评估指标实现，口径与 Python common/metrics.py 对齐。
#include "cxxcommon/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

namespace cxxcommon {
namespace {

// 4-邻接欧氏距离变换（两遍 chamfer，与 scipy 的 EDT 有亚像素级差异，
// 对 HD95/ASSD 的影响远小于模型本身的推理差异）
std::vector<float> DistanceTransform(const std::vector<uint8_t>& mask, int w, int h) {
  const float kInf = 1e9f;
  std::vector<float> d(static_cast<size_t>(w) * h, kInf);
  for (size_t i = 0; i < mask.size(); ++i) {
    if (mask[i]) d[i] = 0.f;
  }
  // forward pass
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      if (d[i] == 0.f) continue;
      float best = kInf;
      if (y > 0) best = std::min(best, d[i - w]);
      if (x > 0) best = std::min(best, d[i - 1]);
      if (y > 0 && x > 0) best = std::min(best, d[i - w - 1]);
      if (y > 0 && x + 1 < w) best = std::min(best, d[i - w + 1]);
      if (best < kInf) d[i] = std::sqrt(best * best + 1.f);
    }
  }
  // backward pass
  for (int y = h - 1; y >= 0; --y) {
    for (int x = w - 1; x >= 0; --x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      if (d[i] == 0.f) continue;
      float best = kInf;
      if (y + 1 < h) best = std::min(best, d[i + w]);
      if (x + 1 < w) best = std::min(best, d[i + 1]);
      if (y + 1 < h && x + 1 < w) best = std::min(best, d[i + w + 1]);
      if (y + 1 < h && x > 0) best = std::min(best, d[i + w - 1]);
      if (best < kInf) d[i] = std::min(d[i], std::sqrt(best * best + 1.f));
    }
  }
  return d;
}

// 边界像素（4-邻域 erosion 的补）
std::vector<uint8_t> SurfaceMask(const std::vector<uint8_t>& mask, int w, int h) {
  std::vector<uint8_t> out(mask.size(), 0);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      if (!mask[i]) continue;
      bool border = false;
      if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        border = true;
      } else {
        border = !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w];
      }
      if (border) out[i] = 1;
    }
  }
  return out;
}

}  // namespace

float Dice(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t n) {
  if (n == 0) return 1.f;
  size_t inter = 0, sa = 0, sb = 0;
  for (size_t i = 0; i < n; ++i) {
    if (a[i]) ++sa;
    if (b[i]) ++sb;
    if (a[i] && b[i]) ++inter;
  }
  const size_t s = sa + sb;
  return s == 0 ? 1.f : static_cast<float>(2.0 * inter / static_cast<double>(s));
}

float Iou(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t n) {
  size_t inter = 0, sa = 0, sb = 0;
  for (size_t i = 0; i < n; ++i) {
    if (a[i]) ++sa;
    if (b[i]) ++sb;
    if (a[i] && b[i]) ++inter;
  }
  const size_t u = sa + sb - inter;
  return u == 0 ? 1.f : static_cast<float>(static_cast<double>(inter) / u);
}

SegMetrics EvaluatePair(const std::vector<uint8_t>& pred, const std::vector<uint8_t>& gt,
                        int w, int h, float boundary_tol) {
  SegMetrics m;
  const size_t n = static_cast<size_t>(w) * h;
  if (pred.size() != n || gt.size() != n) return m;

  size_t sp = 0, sg = 0;
  for (size_t i = 0; i < n; ++i) {
    if (pred[i]) ++sp;
    if (gt[i]) ++sg;
  }
  if (sp == 0 || sg == 0) return m;

  m.dice = Dice(pred, gt, n);
  m.iou = Iou(pred, gt, n);
  m.area_err = sg ? std::fabs(static_cast<float>(sp) - static_cast<float>(sg)) /
                          static_cast<float>(sg)
                  : 0.f;

  // ---- 表面距离
  const std::vector<uint8_t> surf_p = SurfaceMask(pred, w, h);
  const std::vector<uint8_t> surf_g = SurfaceMask(gt, w, h);
  const std::vector<float> dt_p = DistanceTransform(surf_p, w, h);
  const std::vector<float> dt_g = DistanceTransform(surf_g, w, h);

  std::vector<float> d_pg, d_gp;   // pred->gt, gt->pred
  d_pg.reserve(surf_p.size() / 8);
  d_gp.reserve(surf_g.size() / 8);
  for (size_t i = 0; i < n; ++i) {
    if (surf_p[i]) d_pg.push_back(dt_g[i]);
    if (surf_g[i]) d_gp.push_back(dt_p[i]);
  }
  if (d_pg.empty() || d_gp.empty()) return m;

  // ---- HD95
  {
    std::vector<float> all;
    all.reserve(d_pg.size() + d_gp.size());
    all.insert(all.end(), d_pg.begin(), d_pg.end());
    all.insert(all.end(), d_gp.begin(), d_gp.end());
    std::sort(all.begin(), all.end());
    const size_t idx = static_cast<size_t>(0.95 * (all.size() - 1));
    m.hd95 = all[idx];
  }
  // ---- ASSD
  m.assd = (std::accumulate(d_pg.begin(), d_pg.end(), 0.f) / d_pg.size() +
            std::accumulate(d_gp.begin(), d_gp.end(), 0.f) / d_gp.size()) /
           2.f;
  // ---- 边界 F1
  {
    size_t tp_p = 0, tp_g = 0;
    for (float d : d_pg) if (d <= boundary_tol) ++tp_p;
    for (float d : d_gp) if (d <= boundary_tol) ++tp_g;
    const float prec = d_pg.empty() ? 0.f : static_cast<float>(tp_p) / d_pg.size();
    const float rec = d_gp.empty() ? 0.f : static_cast<float>(tp_g) / d_gp.size();
    m.bf1 = (prec + rec) == 0.f ? 0.f : 2.f * prec * rec / (prec + rec);
  }

  m.valid = true;
  return m;
}

std::vector<GtComponent> MaskToComponents(const std::vector<uint8_t>& mask, int w, int h,
                                          int min_area) {
  std::vector<GtComponent> out;
  std::vector<uint8_t> visited(mask.size(), 0);
  std::vector<int> stack;
  for (size_t start = 0; start < mask.size(); ++start) {
    if (!mask[start] || visited[start]) continue;
    GtComponent comp;
    comp.mask.assign(mask.size(), 0);
    stack.clear();
    stack.push_back(static_cast<int>(start));
    visited[start] = 1;
    int minx = w, miny = h, maxx = -1, maxy = -1;
    while (!stack.empty()) {
      const int idx = stack.back();
      stack.pop_back();
      comp.mask[idx] = 1;
      const int y = idx / w, x = idx % w;
      minx = std::min(minx, x); maxx = std::max(maxx, x);
      miny = std::min(miny, y); maxy = std::max(maxy, y);
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (!dx && !dy) continue;
          const int ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
          const int nidx = ny * w + nx;
          if (visited[nidx] || !mask[nidx]) continue;
          visited[nidx] = 1;
          stack.push_back(nidx);
        }
      }
    }
    comp.area = static_cast<int>(std::count(comp.mask.begin(), comp.mask.end(), uint8_t{1}));
    if (comp.area < min_area) continue;
    comp.x1 = static_cast<float>(minx);
    comp.y1 = static_cast<float>(miny);
    comp.x2 = static_cast<float>(maxx + 1);
    comp.y2 = static_cast<float>(maxy + 1);
    out.push_back(std::move(comp));
  }
  return out;
}

Prf ComputePrf(int n_tp, int n_fp, int n_fn) {
  Prf p;
  p.precision = (n_tp + n_fp) ? static_cast<float>(n_tp) / (n_tp + n_fp) : 0.f;
  p.recall = (n_tp + n_fn) ? static_cast<float>(n_tp) / (n_tp + n_fn) : 0.f;
  p.f1 = (p.precision + p.recall) > 0.f
             ? 2.f * p.precision * p.recall / (p.precision + p.recall)
             : 0.f;
  return p;
}

void GreedyMatch(const std::vector<std::vector<uint8_t>>& preds,
                 const std::vector<std::vector<uint8_t>>& gts,
                 float thresh, std::vector<MatchPair>* pairs, int* n_miss, int* n_fp) {
  pairs->clear();
  if (preds.empty() || gts.empty()) {
    *n_miss = static_cast<int>(gts.size());
    *n_fp = static_cast<int>(preds.size());
    return;
  }
  const size_t np = preds.size(), ng = gts.size();
  std::vector<float> iou_mat(np * ng);
  for (size_t i = 0; i < np; ++i) {
    for (size_t j = 0; j < ng; ++j) {
      iou_mat[i * ng + j] = Iou(preds[i], gts[j], preds[i].size());
    }
  }
  std::vector<std::pair<float, std::pair<int, int>>> order;
  order.reserve(np * ng);
  for (size_t i = 0; i < np; ++i) {
    for (size_t j = 0; j < ng; ++j) {
      order.emplace_back(iou_mat[i * ng + j], std::make_pair(static_cast<int>(i),
                                                             static_cast<int>(j)));
    }
  }
  std::sort(order.begin(), order.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<uint8_t> used_p(np, 0), used_g(ng, 0);
  for (const auto& e : order) {
    if (e.first < thresh) break;
    const int i = e.second.first, j = e.second.second;
    if (used_p[i] || used_g[j]) continue;
    used_p[i] = used_g[j] = 1;
    pairs->push_back(MatchPair{i, j, e.first});
  }
  int miss = 0, fp = 0;
  for (size_t j = 0; j < ng; ++j) if (!used_g[j]) ++miss;
  for (size_t i = 0; i < np; ++i) if (!used_p[i]) ++fp;
  *n_miss = miss;
  *n_fp = fp;
}

}  // namespace cxxcommon