// 两阶段流水线：YOLO26n(det) → ROI → MK-UNet → 带类别的目标轮廓。
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cxxcommon/image.hpp"
#include "cxxcommon/mnn_runner.hpp"

namespace twostage {

struct Config {
  std::string det_model;       // yolo26n_det.mnn
  std::string mkunet_model;    // mkunet.mnn
  std::string image_dir;
  std::string gt_dir;
  std::string out_dir;
  std::string list_file;

  int det_imgsz = 512;
  int roi_size = 256;
  float conf = 0.25f;
  float nms_iou = 0.7f;
  int num_classes = 4;
  float pad_ratio = 0.15f;
  float thr = 0.5f;            // MK-UNet 二值化阈值
  int mask_min_area = 50;
  int threads = 4;
  int warmup = 3;
  int max_images = 0;
};

struct Summary {
  int n_images = 0;
  int n_boxes = 0;
  int n_gt = 0;
  int n_pred = 0;
  int n_tp = 0, n_fp = 0, n_fn = 0;

  double dice = 0, iou = 0, hd95 = 0, assd = 0, bf1 = 0, area_err = 0;
  size_t n_matched = 0;

  double det_ms = 0;
  double roi_ms = 0;
  double seg_ms = 0;
  double post_ms = 0;
  double total_ms = 0;
};

Summary RunPipeline(const Config& cfg, bool verbose);
bool RunOne(const Config& cfg, const std::string& image_path, std::string* err);

}  // namespace twostage