// C++ 侧的分割/检测评估指标，口径与 Python 的 common/metrics.py 一致。
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace cxxcommon {

struct SegMetrics {
  float dice = 0.f;
  float iou = 0.f;
  float hd95 = 0.f;
  float assd = 0.f;
  float bf1 = 0.f;
  float area_err = 0.f;
  bool valid = false;
};

SegMetrics EvaluatePair(const std::vector<uint8_t>& pred, const std::vector<uint8_t>& gt,
                       int w, int h, float boundary_tol = 2.0f);

float Dice(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t n);
float Iou(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t n);

// GT mask -> 连通域（每项含 mask/bbox）
struct GtComponent {
  std::vector<uint8_t> mask;
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  int area = 0;
};

std::vector<GtComponent> MaskToComponents(const std::vector<uint8_t>& mask, int w, int h,
                                          int min_area = 50);

struct Prf {
  float precision = 0.f;
  float recall = 0.f;
  float f1 = 0.f;
};

Prf ComputePrf(int n_tp, int n_fp, int n_fn);

// 掩码 IoU 贪心匹配，返回 (匹配对, 未匹配 GT 数, 未匹配预测数)
struct MatchPair {
  int pred_idx = 0;
  int gt_idx = 0;
  float iou = 0.f;
};
void GreedyMatch(const std::vector<std::vector<uint8_t>>& preds,
                 const std::vector<std::vector<uint8_t>>& gts,
                 float thresh, std::vector<MatchPair>* pairs, int* n_miss, int* n_fp);

}  // namespace cxxcommon