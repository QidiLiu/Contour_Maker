// YOLO-Seg 流水线实现：YOLO26n-seg 端到端推理。
#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cxxcommon/eval_harness.hpp"
#include "cxxcommon/metrics.hpp"
#include "cxxcommon/postprocess.hpp"

namespace fs = std::filesystem;
using namespace cxxcommon;

namespace yoloseg {
namespace {

// YOLO26-seg 的 ONNX 导出（ultralytics Segment26 头，export(nms=False)）：
//   output0: (1, N, 6 + nm)  每行 = [x1, y1, x2, y2, score, class_id, coeff_0..nm-1]
//            N = max_det(300)，已按 score 降序（头部自带 TopK），**未做 NMS**
//            坐标是 letterbox 空间的 xyxy
//   output1: (1, nm, Hp, Wp) 掩码原型（Hp=Wp=imgsz/4）
//
// 实测（imgsz=512, nc=4, nm=32）：output0=(1,300,38), output1=(1,32,128,128)。
struct SegHead {
  const float* rows = nullptr;   // (N, cols)
  int N = 0;
  int cols = 0;
  int nm = 0;                    // 掩码系数个数
  const float* protos = nullptr;
  int P = 0, Hp = 0, Wp = 0;
};

bool ParseSegHead(const std::vector<std::vector<int>>& shapes,
                  const std::vector<const float*>& data, int num_classes,
                  SegHead* o, std::string* err) {
  if (shapes.size() < 2) {
    *err = "分割头应输出 2 个张量 (det, protos)，实际 " + std::to_string(shapes.size());
    return false;
  }
  // 4 维的是 protos，3 维的是 det
  int i_p = -1, i_d = -1;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (shapes[i].size() == 4 && i_p < 0) i_p = static_cast<int>(i);
    else if (shapes[i].size() == 3 && i_d < 0) i_d = static_cast<int>(i);
  }
  if (i_p < 0 || i_d < 0) {
    *err = "无法在输出中定位 det(3维) / protos(4维)";
    return false;
  }
  o->P = shapes[i_p][1];
  o->Hp = shapes[i_p][2];
  o->Wp = shapes[i_p][3];
  o->protos = data[i_p];

  // det: (1, N, cols)
  o->N = shapes[i_d][1];
  o->cols = shapes[i_d][2];
  o->rows = data[i_d];
  o->nm = o->P;
  // 列数必须覆盖 4(box)+1(score)+1(class)+nm(coeffs)
  if (o->cols < 6 + o->nm) {
    *err = "det 列数 " + std::to_string(o->cols) + " < 6 + nm(" +
           std::to_string(o->nm) + ")";
    return false;
  }
  return true;
}

std::vector<Instance> Infer(MnnRunner* runner, const Config& cfg, const Image& img,
                            double* infer_ms, std::string* err) {
  std::vector<Instance> out;
  LetterboxInfo lb;
  const Image canvas = Letterbox(img, cfg.imgsz, &lb);

  std::vector<float> input(static_cast<size_t>(cfg.imgsz) * cfg.imgsz * 3);
  // ultralytics 的模型输入是 3 通道 (RGB)。源图为灰度，复制到 3 通道即可
  // （三通道数值相同，因此 BGR/RGB 顺序无影响）。
  const size_t plane = static_cast<size_t>(cfg.imgsz) * cfg.imgsz;
  for (int y = 0; y < cfg.imgsz; ++y) {
    for (int x = 0; x < cfg.imgsz; ++x) {
      const float v = static_cast<float>(canvas.at(y, x)[0]) / 255.f;
      const size_t i = static_cast<size_t>(y) * cfg.imgsz + x;
      input[i] = v;
      input[plane + i] = v;
      input[2 * plane + i] = v;
    }
  }

  if (!runner->Forward(input.data(), {1, 3, cfg.imgsz, cfg.imgsz}, err)) {
    return out;
  }
  if (infer_ms) *infer_ms = runner->LastLatencyMs();

  // MNN 的输出张量按名称排序，顺序稳定；按维度识别 det(3D) / protos(4D)
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

  // ---- 逐行解析: [x1,y1,x2,y2, score, class_id, coeff...]
  // 头部已按 score 降序且带 class_id；仍需按阈值过滤 + NMS（同框会跨类别重复出现）。
  struct Cand {
    Box lb;          // letterbox 空间 xyxy
    Box orig;        // 原图 xyxy
    float score;
    int cls;
    int row;
  };
  std::vector<Cand> cands;
  for (int i = 0; i < head.N; ++i) {
    const float* r = head.rows + static_cast<size_t>(i) * head.cols;
    const float score = r[4];
    if (score < cfg.conf) continue;      // 已降序，但保留 continue 以防导出变化
    const int cls = static_cast<int>(std::lround(r[5]));
    if (cls < 0 || cls >= cfg.num_classes) continue;
    Box lbx{r[0], r[1], r[2], r[3]};
    if (lbx.area() <= 0.f) continue;
    Box ob{(lbx.x1 - lb.pad_x) / lb.ratio, (lbx.y1 - lb.pad_y) / lb.ratio,
           (lbx.x2 - lb.pad_x) / lb.ratio, (lbx.y2 - lb.pad_y) / lb.ratio};
    ob = ClipBox(ob, img.width, img.height);
    if (ob.area() <= 0.f) continue;
    cands.push_back(Cand{lbx, ob, score, cls, i});
  }
  if (cands.empty()) return out;

  // ---- NMS（按类别独立，避免不同类别互相抑制）
  std::vector<Box> nb;
  std::vector<float> ns;
  nb.reserve(cands.size());
  ns.reserve(cands.size());
  for (const auto& c : cands) { nb.push_back(c.orig); ns.push_back(c.score); }
  const std::vector<int> keep = Nms(nb, ns, cfg.nms_iou);

  // ---- 掩码组合 + 贴回原图
  const float proto_scale = static_cast<float>(cfg.imgsz) / head.Wp;  // 512/128 = 4
  for (int k : keep) {
    const Cand& c = cands[k];
    const float* r = head.rows + static_cast<size_t>(c.row) * head.cols + 6;
    Instance inst;
    inst.class_id = c.cls;
    inst.score = c.score;
    inst.box = c.orig;
    inst.mask_w = img.width;
    inst.mask_h = img.height;
    inst.mask.assign(static_cast<size_t>(img.width) * img.height, 0);

    // 原图框 -> proto 网格范围
    const int px1 = std::max(0, static_cast<int>(std::floor(c.lb.x1 / proto_scale)));
    const int py1 = std::max(0, static_cast<int>(std::floor(c.lb.y1 / proto_scale)));
    const int px2 = std::min(head.Wp, static_cast<int>(std::ceil(c.lb.x2 / proto_scale)) + 1);
    const int py2 = std::min(head.Hp, static_cast<int>(std::ceil(c.lb.y2 / proto_scale)) + 1);
    if (px2 <= px1 || py2 <= py1) continue;

    // 在 proto 网格上组合掩码（只算框内区域）
    const int ow = px2 - px1, oh = py2 - py1;
    std::vector<float> crop(static_cast<size_t>(ow) * oh, 0.f);
    for (int p = 0; p < head.P; ++p) {
      const float coef = r[p];
      if (coef == 0.f) continue;
      const float* plane = head.protos + static_cast<size_t>(p) * head.Hp * head.Wp;
      for (int y = 0; y < oh; ++y) {
        const float* src = plane + static_cast<size_t>(py1 + y) * head.Wp + px1;
        float* dst = crop.data() + static_cast<size_t>(y) * ow;
        for (int x = 0; x < ow; ++x) dst[x] += coef * src[x];
      }
    }
    // 掩码在 proto 网格上二值化（ultralytics 用 >0 判定）
    std::vector<uint8_t> crop_bin(static_cast<size_t>(ow) * oh, 0);
    for (size_t i = 0; i < crop_bin.size(); ++i) crop_bin[i] = crop[i] > 0.f ? 1 : 0;

    // 贴回原图：proto 网格 crop -> 原图框区域（最近邻）
    const int dx1 = std::max(0, static_cast<int>(std::floor(c.orig.x1)));
    const int dy1 = std::max(0, static_cast<int>(std::floor(c.orig.y1)));
    const int dx2 = std::min(img.width, static_cast<int>(std::ceil(c.orig.x2)));
    const int dy2 = std::min(img.height, static_cast<int>(std::ceil(c.orig.y2)));
    const int dw = dx2 - dx1, dh = dy2 - dy1;
    if (dw <= 0 || dh <= 0) continue;
    for (int y = 0; y < dh; ++y) {
      const int sy = std::min(oh - 1, static_cast<int>(y * static_cast<float>(oh) / dh));
      uint8_t* dst = inst.mask.data() + static_cast<size_t>(dy1 + y) * img.width + dx1;
      const uint8_t* src = crop_bin.data() + static_cast<size_t>(sy) * ow;
      for (int x = 0; x < dw; ++x) {
        const int sx = std::min(ow - 1, static_cast<int>(x * static_cast<float>(ow) / dw));
        dst[x] = src[sx];
      }
    }

    auto comps = MaskToComponents(inst.mask, img.width, img.height, cfg.mask_min_area);
    if (comps.empty()) continue;
    // 只保留最大连通域（与 Python 侧一致）
    size_t best = 0;
    for (size_t ci = 1; ci < comps.size(); ++ci) {
      if (comps[ci].area > comps[best].area) best = ci;
    }
    inst.mask = comps[best].mask;
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

  // Warm up so the first measured image does not include cold-start/JIT time.
  {
    std::vector<float> warm(static_cast<size_t>(cfg.imgsz) * cfg.imgsz * 3, 0.5f);
    for (int i = 0; i < cfg.warmup; ++i) {
      if (!runner.Forward(warm.data(), {1, 3, cfg.imgsz, cfg.imgsz}, &err)) {
        fprintf(stderr, "[error] warmup failed: %s\n", err.c_str());
        return sum;
      }
    }
  }

  EvalConfig ec;
  ec.image_dir = cfg.image_dir;
  ec.list_file = cfg.list_file;
  ec.gt_dir = cfg.gt_dir;
  ec.gt_instances_dir = cfg.gt_instances_dir;
  ec.out_dir = cfg.out_dir;
  ec.max_images = cfg.max_images;
  ec.min_area = cfg.mask_min_area;
  ec.verbose = verbose;

  const auto infer = [&](const Image& img, std::vector<Instance>* out,
                         double* infer_ms, std::map<std::string, double>* stages,
                         std::string* e) -> bool {
    const auto t0 = std::chrono::steady_clock::now();
    auto insts = Infer(&runner, cfg, img, infer_ms, e);
    const auto t1 = std::chrono::steady_clock::now();
    if (!e->empty()) return false;
    if (stages) {
      (*stages)["preprocess"] +=
          std::chrono::duration<double, std::milli>(t1 - t0).count() - *infer_ms;
    }
    *out = std::move(insts);
    return true;
  };

  const EvalSummary es = RunEvaluation(ec, infer);
  sum.n_images = es.n_images;
  sum.n_gt = es.n_gt;
  sum.n_pred = es.n_pred;
  sum.n_tp = es.n_tp;
  sum.n_fp = es.n_fp;
  sum.n_fn = es.n_fn;
  sum.n_cls_ok = es.n_cls_ok;
  sum.dice = es.dice;
  sum.iou = es.iou;
  sum.hd95 = es.hd95;
  sum.assd = es.assd;
  sum.bf1 = es.bf1;
  sum.area_err = es.area_err;
  sum.n_matched = es.n_matched;
  sum.infer_ms = es.infer_ms;
  sum.post_ms = es.post_ms;
  sum.total_ms = es.total_ms;
  sum.stage_ms = es.stage_ms;
  return sum;
}

}  // namespace yoloseg
