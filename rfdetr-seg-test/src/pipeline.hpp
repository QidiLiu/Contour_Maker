// RF-DETR-Seg-Nano pipeline: single-stage, DETR-style (no NMS).
#pragma once

#include <map>
#include <string>
#include <vector>

#include "cxxcommon/image.hpp"
#include "cxxcommon/mnn_runner.hpp"

namespace rfdetrseg {

struct Config {
  std::string model_path;
  std::string image_dir;
  std::string gt_dir;
  std::string gt_instances_dir;   // ADE20K instance_id,class_id sidecars
  std::string out_dir;
  std::string list_file;

  int resolution = 504;           // must be a multiple of 12
  float conf = 0.25f;
  int num_classes = 4;
  int threads = 4;
  int warmup = 3;
  int max_images = 0;
};

struct Summary {
  int n_images = 0;
  int n_gt = 0;
  int n_pred = 0;
  int n_tp = 0, n_fp = 0, n_fn = 0;
  int n_cls_ok = 0;

  double dice = 0, iou = 0, hd95 = 0, assd = 0, bf1 = 0, area_err = 0;
  size_t n_matched = 0;

  double infer_ms = 0;
  double post_ms = 0;
  double total_ms = 0;
  std::map<std::string, double> stage_ms;
};

Summary RunPipeline(const Config& cfg, bool verbose);
bool RunOne(const Config& cfg, const std::string& image_path, std::string* err);

}  // namespace rfdetrseg
