// two-stage-test 入口：YOLO26n 检测 + MK-UNet ROI 分割（C++ / MNN）。
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "pipeline.hpp"
#include "cxxcommon/metrics.hpp"

namespace {

void PrintUsage() {
  printf(
      "two-stage-test —— 两阶段推理: YOLO26n(det) + MK-UNet (MNN)\n"
      "\n"
      "用法:\n"
      "  two_stage_test --det <yolo26n_det.mnn> --mkunet <mkunet.mnn> \\\n"
      "      --image-dir <dir> [选项]\n"
      "  two_stage_test --det <...> --mkunet <...> --single <image.png>\n"
      "\n"
      "必需:\n"
      "  --det <path>          YOLO26n 检测模型 (.mnn)\n"
      "  --mkunet <path>       MK-UNet 分割模型 (.mnn)\n"
      "  --image-dir <dir>     图像目录（与 --list-file 二选一）\n"
      "\n"
      "可选:\n"
      "  --list-file <path>    图像清单\n"
      "  --single <path>       只推理单张图\n"
      "  --gt-dir <dir>        GT 掩码目录，提供则评估精度\n"
      "  --out-dir <dir>       输出目录\n"
      "  --backend cpu|opencl  推理后端，默认 cpu\n"
      "  --det-imgsz <n>       检测输入尺寸，默认 512\n"
      "  --roi-size <n>        MK-UNet ROI 尺寸，默认 256\n"
      "  --conf <f>            置信度阈值，默认 0.25\n"
      "  --nms-iou <f>         NMS IoU，默认 0.7\n"
      "  --num-classes <n>     类别数，默认 4\n"
      "  --pad-ratio <f>       框外扩比例，默认 0.15\n"
      "  --thr <f>             MK-UNet 二值化阈值，默认 0.5\n"
      "  --min-area <n>        最小连通域面积，默认 50\n"
      "  --threads <n>         CPU 线程数，默认 4\n"
      "  --warmup <n>          预热次数，默认 3\n"
      "  --max-images <n>      最多处理张数，0 = 全部\n"
      "  --quiet               不打印进度\n");
}

}  // namespace

int main(int argc, char** argv) {
  twostage::Config cfg;
  std::string single;
  bool quiet = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "[error] %s 缺少参数\n", name);
        exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      PrintUsage();
      return 0;
    } else if (a == "--det") {
      cfg.det_model = next("--det");
    } else if (a == "--mkunet") {
      cfg.mkunet_model = next("--mkunet");
    } else if (a == "--image-dir") {
      cfg.image_dir = next("--image-dir");
    } else if (a == "--list-file") {
      cfg.list_file = next("--list-file");
    } else if (a == "--single") {
      single = next("--single");
    } else if (a == "--gt-dir") {
      cfg.gt_dir = next("--gt-dir");
    } else if (a == "--out-dir") {
      cfg.out_dir = next("--out-dir");
    } else if (a == "--det-imgsz") {
      cfg.det_imgsz = atoi(next("--det-imgsz").c_str());
    } else if (a == "--roi-size") {
      cfg.roi_size = atoi(next("--roi-size").c_str());
    } else if (a == "--conf") {
      cfg.conf = atof(next("--conf").c_str());
    } else if (a == "--nms-iou") {
      cfg.nms_iou = atof(next("--nms-iou").c_str());
    } else if (a == "--num-classes") {
      cfg.num_classes = atoi(next("--num-classes").c_str());
    } else if (a == "--pad-ratio") {
      cfg.pad_ratio = atof(next("--pad-ratio").c_str());
    } else if (a == "--thr") {
      cfg.thr = atof(next("--thr").c_str());
    } else if (a == "--min-area") {
      cfg.mask_min_area = atoi(next("--min-area").c_str());
    } else if (a == "--threads") {
      cfg.threads = atoi(next("--threads").c_str());
    } else if (a == "--warmup") {
      cfg.warmup = atoi(next("--warmup").c_str());
    } else if (a == "--max-images") {
      cfg.max_images = atoi(next("--max-images").c_str());
    } else if (a == "--backend") {
      const std::string b = next("--backend");
      if (b != "cpu" && b != "opencl") {
        fprintf(stderr, "[error] 未知后端: %s (可选 cpu / opencl)\n", b.c_str());
        return 2;
      }
      // Config 未带 backend 字段（统一走 CPU），此处仅校验参数合法性
    } else if (a == "--quiet") {
      quiet = true;
    } else {
      fprintf(stderr, "[error] 未知参数: %s\n", a.c_str());
      PrintUsage();
      return 2;
    }
  }

  if (cfg.det_model.empty() || cfg.mkunet_model.empty()) {
    fprintf(stderr, "[error] 必须同时指定 --det 与 --mkunet\n\n");
    PrintUsage();
    return 2;
  }

  if (!single.empty()) {
    std::string err;
    if (!twostage::RunOne(cfg, single, &err)) {
      fprintf(stderr, "[error] %s\n", err.c_str());
      return 1;
    }
    return 0;
  }
  if (cfg.image_dir.empty() && cfg.list_file.empty()) {
    fprintf(stderr, "[error] 必须指定 --image-dir 或 --list-file\n\n");
    PrintUsage();
    return 2;
  }

  printf("\n");
  printf("==============================================================\n");
  printf("=== two-stage-test: YOLO26n(det) + MK-UNet (C++ / MNN)\n");
  printf("==============================================================\n\n");

  const auto s = twostage::RunPipeline(cfg, !quiet);
  const double n = s.n_images > 0 ? static_cast<double>(s.n_images) : 1.0;

  printf("\n[汇总]\n");
  printf("  images        %d\n", s.n_images);
  printf("  GT / pred     %d / %d\n", s.n_gt, s.n_pred);
  if (s.n_gt > 0) {
    const auto prf = cxxcommon::ComputePrf(s.n_tp, s.n_fp, s.n_fn);
    printf("  TP/FP/FN      %d / %d / %d\n", s.n_tp, s.n_fp, s.n_fn);
    printf("  P/R/F1        %.4f / %.4f / %.4f\n", prf.precision, prf.recall, prf.f1);
  }
  if (s.n_matched) {
    printf("\n[分割指标]  n_matched=%zu\n", s.n_matched);
    printf("  Dice      %.4f\n", s.dice);
    printf("  IoU       %.4f\n", s.iou);
    printf("  HD95      %.2f px\n", s.hd95);
    printf("  ASSD      %.2f px\n", s.assd);
    printf("  BF1       %.4f\n", s.bf1);
    printf("  area_err  %.4f\n", s.area_err);
  }
  printf("\n[速度]\n");
  printf("  检测       %7.2f ms/图\n", s.det_ms / n);
  printf("  ROI 裁剪   %7.2f ms/图\n", s.roi_ms / n);
  printf("  MK-UNet    %7.2f ms/图\n", s.seg_ms / n);
  printf("  后处理     %7.2f ms/图\n", s.post_ms / n);
  printf("  合计       %7.2f ms/图  (%.1f FPS)\n", s.total_ms / n,
         1000.0 * n / std::max(1e-9, s.total_ms));
  printf("\n");
  return 0;
}