// one-stage-test 入口：YOLO26n-seg 端到端推理（C++ / MNN）。
#include <cstdio>
#include <cstring>
#include <string>

#include "pipeline.hpp"
#include "cxxcommon/metrics.hpp"

namespace {

void PrintUsage() {
  printf(
      "one-stage-test —— 单阶段 YOLO26n-seg 端到端推理 (MNN)\n"
      "\n"
      "用法:\n"
      "  one-stage_test --model <yolo26n_seg.mnn> --image-dir <dir> [选项]\n"
      "  one-stage_test --model <...> --single <image.png>\n"
      "\n"
      "必需:\n"
      "  --model <path>        MNN 模型路径\n"
      "  --image-dir <dir>     图像目录（与 --list-file 二选一）\n"
      "\n"
      "可选:\n"
      "  --list-file <path>    图像清单，每行一个路径\n"
      "  --single <path>       只推理单张图并打印实例详情\n"
      "  --gt-dir <dir>        GT 掩码目录，提供则评估精度\n"
      "  --out-dir <dir>       输出目录（per_image.csv）\n"
      "  --backend cpu|opencl  推理后端，默认 cpu（opencl 不可用时自动降级）\n"
      "  --imgsz <n>           推理输入尺寸，默认 512\n"
      "  --conf <f>            置信度阈值，默认 0.25\n"
      "  --nms-iou <f>         NMS IoU 阈值，默认 0.7\n"
      "  --num-classes <n>     类别数，默认 4\n"
      "  --min-area <n>        掩码最小连通域面积，默认 50\n"
      "  --threads <n>         CPU 线程数，默认 4\n"
      "  --warmup <n>          预热次数，默认 3\n"
      "  --max-images <n>      最多处理多少张，0 = 全部\n"
      "  --quiet               不打印进度\n"
      "\n"
      "示例:\n"
      "  one-stage_test --model models/yolo26n_seg.mnn \\\n"
      "      --image-dir data/unified/images \\\n"
      "      --gt-dir data/unified/masks \\\n"
      "      --list-file splits/test.txt --out-dir reports/one_stage_cxx\n");
}

}  // namespace

int main(int argc, char** argv) {
  onestage::Config cfg;
  std::string single;
  bool quiet = false;
  cxxcommon::Backend backend = cxxcommon::Backend::kCpu;
  bool backend_set = false;

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
    } else if (a == "--model") {
      cfg.model_path = next("--model");
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
    } else if (a == "--imgsz") {
      cfg.imgsz = atoi(next("--imgsz").c_str());
    } else if (a == "--conf") {
      cfg.conf = atof(next("--conf").c_str());
    } else if (a == "--nms-iou") {
      cfg.nms_iou = atof(next("--nms-iou").c_str());
    } else if (a == "--num-classes") {
      cfg.num_classes = atoi(next("--num-classes").c_str());
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
      if (b == "opencl") {
        backend = cxxcommon::Backend::kOpenCl;
      } else if (b == "cpu") {
        backend = cxxcommon::Backend::kCpu;
      } else {
        fprintf(stderr, "[error] 未知后端: %s (可选 cpu / opencl)\n", b.c_str());
        return 2;
      }
      backend_set = true;
    } else if (a == "--quiet") {
      quiet = true;
    } else {
      fprintf(stderr, "[error] 未知参数: %s\n", a.c_str());
      PrintUsage();
      return 2;
    }
  }

  if (cfg.model_path.empty()) {
    fprintf(stderr, "[error] 必须指定 --model\n\n");
    PrintUsage();
    return 2;
  }

  if (!single.empty()) {
    std::string err;
    if (!onestage::RunOne(cfg, single, &err)) {
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
  printf("=== one-stage-test: YOLO26n-seg 端到端 (C++ / MNN)\n");
  printf("==============================================================\n\n");

  const auto s = onestage::RunPipeline(cfg, !quiet);

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
  printf("  推理       %7.2f ms/图\n", s.infer_ms / n);
  printf("  后处理     %7.2f ms/图\n", s.post_ms / n);
  printf("  合计       %7.2f ms/图  (%.1f FPS)\n", s.total_ms / n,
         1000.0 * n / std::max(1e-9, s.total_ms));
  printf("\n");

  (void)backend;
  (void)backend_set;
  return 0;
}