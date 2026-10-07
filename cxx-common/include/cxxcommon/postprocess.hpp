// YOLO 输出后处理：NMS、框解码、YOLACT 式掩码组合。
// 口径与 ultralytics 默认配置一致（详见 common/config.py）。
#pragma once

#include <vector>

#include "cxxcommon/image.hpp"

namespace cxxcommon {

struct YoloDetOut {
  std::vector<Box> boxes;     // 已反 letterbox 变换，原图坐标
  std::vector<int> class_ids;
  std::vector<float> scores;
  // NMS 保留的**原始输出索引**（按分数降序），用于把 protos 的每实例系数
  // 对齐到解码后的实例顺序。
  std::vector<int> indices;
};

// 标准贪心 NMS（boxes 为 xyxy，已在原图坐标系）
std::vector<int> Nms(const std::vector<Box>& boxes, const std::vector<float>& scores,
                     float iou_thresh);

// 从 YOLO 检测输出解码框：
//   boxes: (N, 4) 中心点 cx,cy,w,h（letterbox 坐标系，连续内存）
//   scores: (N,)   最大类别得分
//   classes: (N,)  argmax 类别
void DecodeDetections(const float* boxes, const float* scores, const int* classes,
                      int count, float conf_thresh, int orig_w, int orig_h,
                      float ratio, int pad_x, int pad_y, float nms_iou,
                      YoloDetOut* out);

// box 索引集合：NMS 保留的原始下标（按分数降序），用于把 protos 的系数对齐到实例
std::vector<int> NmsIndices(const std::vector<Box>& boxes, const float* scores,
                            int count, float iou_thresh, float conf_thresh);

// YOLACT 式掩码组合：
//   protos: (P*H*W,)      mask prototype，展平
//   coeffs: (N*P,)        每个实例的 P 维系数
//   P, H, W:              prototype 通道数与空间尺寸（letterbox 空间）
//   将 coeffs · protos 组合成 (N,H,W)，裁剪到各自框内，再上采样回原图。
std::vector<std::vector<uint8_t>> CombineMasks(
    const float* protos, int P, int H, int W,
    const float* coeffs, const std::vector<Box>& boxes_lb,
    int lb_size, int orig_w, int orig_h, float ratio, int pad_x, int pad_y);

// 二值化 + 最小连通域面积过滤
void FilterMasks(std::vector<std::vector<uint8_t>>* masks, int w, int h, int min_area);

}  // namespace cxxcommon