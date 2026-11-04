// Shared evaluation harness: image listing, GT decoding, matching, metrics and
// per-image CSV output.  All scheme binaries differ only in their inference
// callback, which keeps the Python and C++ evaluation protocols identical.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "cxxcommon/image.hpp"

namespace cxxcommon {

struct EvalConfig {
  std::string image_dir;
  std::string list_file;
  std::string gt_dir;            // GT mask directory (binary or instance-id PNG)
  std::string gt_instances_dir;  // optional: <uid>.csv sidecars (instance_id,class_id)
  std::string out_dir;           // optional: writes per_image.csv
  int max_images = 0;            // 0 = all
  int min_area = 50;
  float match_iou = 0.5f;
  float boundary_tol = 2.0f;
  bool verbose = true;
};

struct EvalSummary {
  int n_images = 0;
  int n_gt = 0;
  int n_pred = 0;
  int n_tp = 0;
  int n_fp = 0;
  int n_fn = 0;
  int n_cls_ok = 0;   // matched pairs with equal class ids

  double dice = 0, iou = 0, hd95 = 0, assd = 0, bf1 = 0, area_err = 0;
  size_t n_matched = 0;

  double infer_ms = 0;
  double post_ms = 0;
  double total_ms = 0;
  std::map<std::string, double> stage_ms;
};

// Returns false on unrecoverable inference errors for this image.
using InferFn = std::function<bool(const Image&, std::vector<Instance>*,
                                   double* infer_ms,
                                   std::map<std::string, double>* stages,
                                   std::string* err)>;

EvalSummary RunEvaluation(const EvalConfig& cfg, const InferFn& infer);

// Decode a GT mask into binary instances.  When a sidecar exists the mask is an
// instance-id map and each id carries its class from the sidecar.
std::vector<Instance> LoadGtInstances(const Image& idmap, const std::string& sidecar,
                                      int min_area);

}  // namespace cxxcommon
