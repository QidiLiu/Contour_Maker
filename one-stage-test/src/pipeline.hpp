// 单阶段流水线：YOLO26n-seg 端到端推理。
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cxxcommon/image.hpp"
#include "cxxcommon/mnn_runner.hpp"

namespace onestage {

struct Config {
  std::string model_path;
  std::string image_dir;
  std::string gt_dir;          // 可选，有则做精度评估
  std::string out_dir;         // 可选，输出 per_image.csv
  std::string list_file;       // 可选，待处理图像清单
  int imgsz = 512;
  float conf = 0.25f;
  float nms_iou = 0.7f;
  int num_classes = 4;
  int mask_min_area = 50;
  int threads = 4;
  int warmup = 3;
  int max_images = 0;          // 0 = 全部
};

struct Summary {
  int n_images = 0;
  int n_gt = 0;
  int n_pred = 0;
  int n_tp = 0;
  int n_fp = 0;
  int n_fn = 0;

  double dice = 0, iou = 0, hd95 = 0, assd = 0, bf1 = 0, area_err = 0;
  size_t n_matched = 0;

  double infer_ms = 0;
  double post_ms = 0;
  double total_ms = 0;
};

// 跑完整批，返回汇总
Summary RunPipeline(const Config& cfg, bool verbose);

// 单图推理（测试/调试用）
bool RunOne(const Config& cfg, const std::string& image_path, std::string* err);

}  // namespace onestage