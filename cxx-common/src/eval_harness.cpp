// Shared evaluation harness implementation.
#include "cxxcommon/eval_harness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "cxxcommon/metrics.hpp"

namespace fs = std::filesystem;

namespace cxxcommon {
namespace {

std::map<int, int> ReadSidecar(const std::string& path) {
  std::map<int, int> out;
  std::ifstream f(path);
  if (!f) return out;
  std::string line;
  std::getline(f, line);  // header: instance_id,class_id
  while (std::getline(f, line)) {
    const size_t comma = line.find(',');
    if (comma == std::string::npos) continue;
    try {
      out[std::stoi(line.substr(0, comma))] = std::stoi(line.substr(comma + 1));
    } catch (...) {
      // Ignore malformed rows.
    }
  }
  return out;
}

std::vector<std::string> ListImages(const EvalConfig& cfg) {
  std::vector<std::string> images;
  if (!cfg.list_file.empty()) {
    std::ifstream f(cfg.list_file);
    std::string line;
    while (std::getline(f, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
      if (!line.empty()) images.push_back(line);
    }
  } else if (!cfg.image_dir.empty()) {
    for (const auto& e : fs::directory_iterator(cfg.image_dir)) {
      if (!e.is_regular_file()) continue;
      const std::string ext = e.path().extension().string();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") images.push_back(e.path().string());
    }
  }
  std::sort(images.begin(), images.end());
  if (cfg.max_images > 0 && static_cast<int>(images.size()) > cfg.max_images) {
    images.resize(cfg.max_images);
  }
  return images;
}

}  // namespace

std::vector<Instance> LoadGtInstances(const Image& idmap, const std::string& sidecar,
                                      int min_area) {
  std::vector<Instance> out;
  const int w = idmap.width, h = idmap.height;
  if (w <= 0 || h <= 0) return out;

  const auto id_to_class = sidecar.empty() ? std::map<int, int>{} : ReadSidecar(sidecar);

  if (id_to_class.empty()) {
    // Binary mask convention (ultrasound).
    std::vector<uint8_t> binary(idmap.data.size(), 0);
    for (size_t i = 0; i < binary.size(); ++i) binary[i] = idmap.data[i] > 127 ? 1 : 0;
    for (auto& comp : MaskToComponents(binary, w, h, min_area)) {
      Instance inst;
      inst.mask = std::move(comp.mask);
      inst.mask_w = w;
      inst.mask_h = h;
      inst.class_id = -1;
      inst.has_mask = true;
      out.push_back(std::move(inst));
    }
    return out;
  }

  std::map<int, std::vector<uint8_t>> per_id;
  for (size_t i = 0; i < idmap.data.size(); ++i) {
    const int v = idmap.data[i];
    if (v == 0) continue;
    auto it = per_id.find(v);
    if (it == per_id.end()) {
      it = per_id.emplace(v, std::vector<uint8_t>(idmap.data.size(), 0)).first;
    }
    it->second[i] = 1;
  }
  for (auto& [id, mask] : per_id) {
    const auto cls_it = id_to_class.find(id);
    const int cls = cls_it == id_to_class.end() ? -1 : cls_it->second;
    for (auto& comp : MaskToComponents(mask, w, h, min_area)) {
      Instance inst;
      inst.mask = std::move(comp.mask);
      inst.mask_w = w;
      inst.mask_h = h;
      inst.class_id = cls;
      inst.has_mask = true;
      out.push_back(std::move(inst));
    }
  }
  return out;
}

EvalSummary RunEvaluation(const EvalConfig& cfg, const InferFn& infer) {
  EvalSummary sum;
  std::string err;
  const std::vector<std::string> images = ListImages(cfg);
  printf("[data] %zu images\n", images.size());
  if (images.empty()) return sum;
  if (!cfg.out_dir.empty()) {
    std::error_code ec;
    fs::create_directories(cfg.out_dir, ec);
  }

  double dice_sum = 0, iou_sum = 0, hd_sum = 0, asd_sum = 0, bf1_sum = 0, ae_sum = 0;
  std::string csv = "image,n_gt,n_pred,n_miss,n_fp\n";

  for (size_t idx = 0; idx < images.size(); ++idx) {
    Image img;
    if (!LoadImage(images[idx], &img, &err)) {
      fprintf(stderr, "[warn] %s\n", err.c_str());
      continue;
    }
    const int W = img.width, H = img.height;

    std::vector<Instance> preds;
    double infer_ms = 0;
    std::map<std::string, double> stages;
    const auto t0 = std::chrono::steady_clock::now();
    if (!infer(img, &preds, &infer_ms, &stages, &err)) {
      fprintf(stderr, "[error] %s: %s\n", images[idx].c_str(), err.c_str());
      err.clear();
      continue;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    sum.infer_ms += infer_ms;
    sum.post_ms += total_ms - infer_ms;
    sum.total_ms += total_ms;
    for (const auto& [k, v] : stages) sum.stage_ms[k] += v;
    ++sum.n_images;
    sum.n_pred += static_cast<int>(preds.size());

    std::vector<Instance> gts;
    if (!cfg.gt_dir.empty()) {
      const std::string stem = fs::path(images[idx]).stem().string();
      Image gt;
      if (LoadImage((fs::path(cfg.gt_dir) / (stem + ".png")).string(), &gt, nullptr) &&
          gt.width == W && gt.height == H) {
        std::string sidecar;
        if (!cfg.gt_instances_dir.empty()) {
          sidecar = (fs::path(cfg.gt_instances_dir) / (stem + ".csv")).string();
        }
        gts = LoadGtInstances(gt, sidecar, cfg.min_area);
      }
    }
    sum.n_gt += static_cast<int>(gts.size());

    if (!gts.empty()) {
      std::vector<std::vector<uint8_t>> pm, gm;
      pm.reserve(preds.size());
      gm.reserve(gts.size());
      for (const auto& p : preds) pm.push_back(p.mask);
      for (const auto& g : gts) gm.push_back(g.mask);
      std::vector<MatchPair> pairs;
      int miss = 0, fp = 0;
      GreedyMatch(pm, gm, cfg.match_iou, &pairs, &miss, &fp);
      sum.n_tp += static_cast<int>(pairs.size());
      sum.n_fn += miss;
      sum.n_fp += fp;
      char buf[320];
      snprintf(buf, sizeof(buf), "%s,%d,%zu,%d,%d\n",
               fs::path(images[idx]).filename().string().c_str(),
               static_cast<int>(gts.size()), pm.size(), miss, fp);
      csv += buf;

      for (const auto& p : pairs) {
        if (preds[p.pred_idx].class_id >= 0 && gts[p.gt_idx].class_id >= 0 &&
            preds[p.pred_idx].class_id == gts[p.gt_idx].class_id) {
          ++sum.n_cls_ok;
        }
        const auto m = EvaluatePair(pm[p.pred_idx], gm[p.gt_idx], W, H, cfg.boundary_tol);
        if (!m.valid) continue;
        dice_sum += m.dice;
        iou_sum += m.iou;
        hd_sum += m.hd95;
        asd_sum += m.assd;
        bf1_sum += m.bf1;
        ae_sum += m.area_err;
        ++sum.n_matched;
      }
    }

    if (cfg.verbose && (idx + 1) % 50 == 0) {
      printf("  %zu/%zu  %.2f ms/img\n", idx + 1, images.size(),
             sum.total_ms / (idx + 1));
      fflush(stdout);
    }
  }

  if (sum.n_matched) {
    const double n = static_cast<double>(sum.n_matched);
    sum.dice = dice_sum / n;
    sum.iou = iou_sum / n;
    sum.hd95 = hd_sum / n;
    sum.assd = asd_sum / n;
    sum.bf1 = bf1_sum / n;
    sum.area_err = ae_sum / n;
  }
  if (!cfg.out_dir.empty()) {
    std::ofstream f(cfg.out_dir + "/per_image.csv");
    f << csv;
    std::ofstream js(cfg.out_dir + "/summary.json");
    const double n = sum.n_images > 0 ? static_cast<double>(sum.n_images) : 1.0;
    js << "{\n"
       << "  \"n_images\": " << sum.n_images << ",\n"
       << "  \"segmentation\": {\"n_matched\": " << sum.n_matched
       << ", \"dice\": " << sum.dice << ", \"iou\": " << sum.iou
       << ", \"hd95\": " << sum.hd95 << ", \"assd\": " << sum.assd
       << ", \"bf1\": " << sum.bf1 << ", \"area_err\": " << sum.area_err
       << (cfg.gt_instances_dir.empty()
               ? std::string("")
               : ", \"cls_acc\": " + std::to_string(
                     sum.n_matched ? static_cast<double>(sum.n_cls_ok) / sum.n_matched : 0.0))
       << "},\n"
       << "  \"detection\": {\"gt\": " << sum.n_gt << ", \"pred\": " << sum.n_pred
       << ", \"tp\": " << sum.n_tp << ", \"fp\": " << sum.n_fp << ", \"fn\": " << sum.n_fn
       << "},\n"
       << "  \"speed\": {\"total_ms\": " << (sum.total_ms / n)
       << ", \"infer_ms\": " << (sum.infer_ms / n)
       << ", \"post_ms\": " << (sum.post_ms / n)
       << ", \"fps\": " << (1000.0 * n / std::max(1e-9, sum.total_ms)) << "}\n"
       << "}\n";
  }
  return sum;
}

}  // namespace cxxcommon
