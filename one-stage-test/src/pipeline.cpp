// 单阶段流水线实现：YOLO26n-seg 端到端推理。
#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cxxcommon/metrics.hpp"
#include "cxxcommon/postprocess.hpp"

namespace fs = std::filesystem;
using namespace cxxcommon;

namespace onestage {
namespace {

// ultralytics 以 nms=False 导出时的分割头输出（3 个张量）：
//   out0: (1, 4+nc, N)  boxes(cx,cy,w,h) + per-class scores（CHW）
//   out1: (1, P, h, w)  mask prototypes
//   out2: (1, P, N)     per-instance mask coefficients
struct SegHead {
  std::vector<float> boxes;    // (N,4) cx,cy,w,h，已转成 NCHW 布局
  std::vector<float> scores;   // (N,)  最大类别得分
  std::vector<int> classes;    // (N,)  argmax 类别
  const float* protos = nullptr;
  const float* coeffs = nullptr;
  int P = 0, Hp = 0, Wp = 0;
  int N = 0;
};

bool ParseSegHead(const std::vector<std::vector<int>>& shapes,
                  const std::vector<const float*>& data, int num_classes,
                  SegHead* o, std::string* err) {
  if (shapes.size() < 3) {
    *err = "分割头应输出 3 个张量(det/protos/coeffs)，实际 " +
           std::to_string(shapes.size()) +
           "。请确认 ONNX 由 one-stage-train/export_onnx.py 以 nms=False 导出。";
    return false;
  }
  // 按形状识别，而不是依赖输出顺序：
  //   protos : 4 维 (1,P,h,w)
  //   coeffs : 3 维 (1,P,N)
  //   det    : 3 维 (1,4+nc,N)
  int i_protos = -1, i_coeffs = -1, i_det = -1;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& s = shapes[i];
    if (s.size() == 4 && i_protos < 0) {
      i_protos = static_cast<int>(i);
    } else if (s.size() == 3) {
      const int c = s[1];
      if (c >= 4 + num_classes && i_det < 0) {
        i_det = static_cast<int>(i);
      } else if (i_coeffs < 0) {
        i_coeffs = static_cast<int>(i);
      }
    }
  }
  if (i_protos < 0 || i_coeffs < 0 || i_det < 0) {
    *err = "无法在输出中定位 det/protos/coeffs";
    return false;
  }

  const auto& ps = shapes[i_protos];
  o->P = ps[1];
  o->Hp = ps[2];
  o->Wp = ps[3];
  o->protos = data[i_protos];

  const auto& cs = shapes[i_coeffs];
  o->N = cs[2];
  o->coeffs = data[i_coeffs];

  const auto& ds = shapes[i_det];
  const int C = ds[1];
  const int N = ds[2];
  if (N != o->N) {
    *err = "det 的 N(" + std::to_string(N) + ") 与 coeffs 的 N(" +
           std::to_string(o->N) + ") 不一致";
    return false;
  }

  const float* det = data[i_det];
  o->boxes.resize(static_cast<size_t>(N) * 4);
  o->scores.resize(N);
  o->classes.resize(N);
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < 4; ++k) {
      o->boxes[static_cast<size_t>(i) * 4 + k] = det[static_cast<size_t>(k) * N + i];
    }
    int best = 0;
    float bestv = det[static_cast<size_t>(4) * N + i];
    for (int c = 1; c < num_classes; ++c) {
      const float v = det[static_cast<size_t>(4 + c) * N + i];
      if (v > bestv) {
        bestv = v;
        best = c;
      }
    }
    o->classes[i] = best;
    o->scores[i] = bestv;
  }
  return true;
}

std::vector<Instance> Infer(MnnRunner* runner, const Config& cfg, const Image& img,
                            double* infer_ms, std::string* err) {
  std::vector<Instance> out;
  LetterboxInfo lb;
  const Image canvas = Letterbox(img, cfg.imgsz, &lb);

  std::vector<float> input(static_cast<size_t>(cfg.imgsz) * cfg.imgsz);
  for (int y = 0; y < cfg.imgsz; ++y) {
    for (int x = 0; x < cfg.imgsz; ++x) {
      input[static_cast<size_t>(y) * cfg.imgsz + x] =
          static_cast<float>(canvas.at(y, x)[0]) / 255.f;
    }
  }

  if (!runner->Forward(input.data(), {1, 1, cfg.imgsz, cfg.imgsz}, err)) {
    return out;
  }
  if (infer_ms) *infer_ms = runner->LastLatencyMs();

  // MNN 的输出张量按名称排序，顺序稳定；这里按形状识别 det/protos/coeffs
  std::vector<std::vector<int>> shapes(runner->NumOutputs());
  std::vector<const float*> datas(runner->NumOutputs(), nullptr);
  for (int i = 0; i < runner->NumOutputs(); ++i) {
    if (!runner->Output(i, &datas[i], &shapes[i])) {
      *err = "取输出张量失败: " + runner->OutputName(i);
      return out;
    }
  }

  SegHead head;
  if (!ParseSegHead(shapes, datas, cfg.num_classes, &head, err)) return out;
  if (head.N == 0) return out;

  // ---- 置信度筛选 + NMS（框解码到原图坐标）
  YoloDetOut det;
  DecodeDetections(head.boxes.data(), head.scores.data(), head.classes.data(), head.N,
                   cfg.conf, img.width, img.height, lb.ratio, lb.pad_x,
                   lb.pad_y, cfg.nms_iou, &det);
  if (det.boxes.empty()) return out;

  // ---- 掩码组合：需要 letterbox 空间的框 + 与 NMS 顺序对齐的每实例系数
  std::vector<Box> lb_boxes;
  lb_boxes.reserve(det.boxes.size());
  for (const auto& b : det.boxes) {
    lb_boxes.push_back(Box{b.x1 * lb.ratio + lb.pad_x,
                           b.y1 * lb.ratio + lb.pad_y,
                           b.x2 * lb.ratio + lb.pad_x,
                           b.y2 * lb.ratio + lb.pad_y});
  }
  std::vector<float> coeffs;
  coeffs.reserve(det.boxes.size() * head.P);
  for (int idx : det.indices) {
    const float* c = head.coeffs + static_cast<size_t>(idx) * head.P;
    coeffs.insert(coeffs.end(), c, c + head.P);
  }

  auto masks = CombineMasks(head.protos, head.P, head.Hp, head.Wp, coeffs.data(),
                            lb_boxes, cfg.imgsz, img.width, img.height,
                            lb.ratio, lb.pad_x, lb.pad_y);
  FilterMasks(&masks, img.width, img.height, cfg.mask_min_area);

  for (size_t i = 0; i < det.boxes.size() && i < masks.size(); ++i) {
    Instance inst;
    inst.class_id = det.class_ids[i];
    inst.score = det.scores[i];
    inst.box = det.boxes[i];
    inst.mask = masks[i];
    inst.mask_w = img.width;
    inst.mask_h = img.height;
    inst.has_mask = true;
    if (MaskToContour(inst.mask, img.width, img.height, &inst.contour)) {
      out.push_back(std::move(inst));
    }
  }
  return out;
}

}  // namespace

bool RunOne(const Config& cfg, const std::string& image_path, std::string* err) {
  MnnRunner runner;
  Backend actual = Backend::kCpu;
  if (!runner.Load(cfg.model_path, Backend::kCpu, cfg.threads, &actual, err)) {
    return false;
  }
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
  printf("[model] %s  backend=%s threads=%d imgsz=%d conf=%.2f\n",
         cfg.model_path.c_str(), BackendName(actual), cfg.threads, cfg.imgsz,
         cfg.conf);

  // ---- 图像清单
  std::vector<std::string> images;
  if (!cfg.list_file.empty()) {
    std::ifstream f(cfg.list_file);
    std::string line;
    while (std::getline(f, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
      }
      if (!line.empty()) images.push_back(line);
    }
  } else if (!cfg.image_dir.empty()) {
    for (const auto& e : fs::directory_iterator(cfg.image_dir)) {
      if (!e.is_regular_file()) continue;
      const std::string ext = e.path().extension().string();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        images.push_back(e.path().string());
      }
    }
  }
  std::sort(images.begin(), images.end());
  if (cfg.max_images > 0 && static_cast<int>(images.size()) > cfg.max_images) {
    images.resize(cfg.max_images);
  }
  printf("[data] %zu images\n", images.size());
  if (images.empty()) return sum;

  if (!cfg.out_dir.empty()) fs::create_directories(cfg.out_dir);

  // ---- 预热（缓存冷启动 / JIT 预编译）
  {
    std::vector<float> warm(static_cast<size_t>(cfg.imgsz) * cfg.imgsz, 0.5f);
    for (int i = 0; i < cfg.warmup; ++i) {
      if (!runner.Forward(warm.data(), {1, 1, cfg.imgsz, cfg.imgsz}, &err)) {
        fprintf(stderr, "[error] 预热失败: %s\n", err.c_str());
        return sum;
      }
    }
  }

  double dice_sum = 0, iou_sum = 0, hd_sum = 0, asd_sum = 0, bf1_sum = 0, ae_sum = 0;
  size_t n_matched = 0;
  std::string csv = "image,n_gt,n_pred,n_miss,n_fp\n";

  for (size_t idx = 0; idx < images.size(); ++idx) {
    const std::string& ip = images[idx];
    Image img;
    if (!LoadImage(ip, &img, &err)) {
      fprintf(stderr, "[warn] %s\n", err.c_str());
      continue;
    }
    const int W = img.width, H = img.height;

    double infer_ms = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto insts = Infer(&runner, cfg, img, &infer_ms, &err);
    const auto t1 = std::chrono::steady_clock::now();
    if (!err.empty()) {
      fprintf(stderr, "[error] %s: %s\n", ip.c_str(), err.c_str());
      err.clear();
      continue;
    }
    sum.infer_ms += infer_ms;
    sum.post_ms += std::chrono::duration<double, std::milli>(t1 - t0).count() - infer_ms;
    sum.total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ++sum.n_images;
    sum.n_pred += static_cast<int>(insts.size());

    // ---- GT
    std::vector<std::vector<uint8_t>> gt_masks;
    if (!cfg.gt_dir.empty()) {
      const std::string stem = fs::path(ip).stem().string();
      const std::string gp = (fs::path(cfg.gt_dir) / (stem + ".png")).string();
      Image gt;
      if (LoadImage(gp, &gt, nullptr) && gt.width == W && gt.height == H) {
        auto comps = MaskToComponents(gt.data, W, H, cfg.mask_min_area);
        for (auto& c : comps) gt_masks.push_back(std::move(c.mask));
      }
    }
    sum.n_gt += static_cast<int>(gt_masks.size());

    if (!gt_masks.empty()) {
      std::vector<std::vector<uint8_t>> pm;
      pm.reserve(insts.size());
      for (const auto& in : insts) pm.push_back(in.mask);
      std::vector<MatchPair> pairs;
      int miss = 0, fp = 0;
      GreedyMatch(pm, gt_masks, 0.5f, &pairs, &miss, &fp);
      sum.n_tp += static_cast<int>(pairs.size());
      sum.n_fn += miss;
      sum.n_fp += fp;
      char buf[320];
      snprintf(buf, sizeof(buf), "%s,%d,%zu,%d,%d\n",
               fs::path(ip).filename().string().c_str(),
               static_cast<int>(gt_masks.size()), pm.size(), miss, fp);
      csv += buf;

      for (const auto& p : pairs) {
        const auto m = EvaluatePair(pm[p.pred_idx], gt_masks[p.gt_idx], W, H);
        if (!m.valid) continue;
        dice_sum += m.dice;
        iou_sum += m.iou;
        hd_sum += m.hd95;
        asd_sum += m.assd;
        bf1_sum += m.bf1;
        ae_sum += m.area_err;
        ++n_matched;
      }
    }

    if (verbose && (idx + 1) % 50 == 0) {
      printf("  %zu/%zu  %.2f ms/图\n", idx + 1, images.size(),
             sum.total_ms / (idx + 1));
      fflush(stdout);
    }
  }

  if (n_matched) {
    sum.dice = dice_sum / n_matched;
    sum.iou = iou_sum / n_matched;
    sum.hd95 = hd_sum / n_matched;
    sum.assd = asd_sum / n_matched;
    sum.bf1 = bf1_sum / n_matched;
    sum.area_err = ae_sum / n_matched;
  }
  sum.n_matched = n_matched;

  if (!cfg.out_dir.empty()) {
    std::ofstream f(cfg.out_dir + "/per_image.csv");
    f << csv;
  }

  return sum;
}

}  // namespace onestage