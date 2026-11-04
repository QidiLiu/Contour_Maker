// rfdetr-seg-test entry point: RF-DETR-Seg-Nano inference (C++ / MNN).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "cxxcommon/metrics.hpp"
#include "pipeline.hpp"

namespace {

void PrintUsage() {
  printf(
      "rfdetr-seg-test -- RF-DETR-Seg-Nano end-to-end inference (MNN)\n"
      "\n"
      "Usage:\n"
      "  rfdetr_seg_test --model <rfdetr_seg_nano.mnn> --image-dir <dir> [options]\n"
      "  rfdetr_seg_test --model <...> --single <image.png>\n"
      "\n"
      "Required:\n"
      "  --model <path>          MNN model path\n"
      "  --image-dir <dir>       image directory (or --list-file)\n"
      "\n"
      "Options:\n"
      "  --list-file <path>      one image path per line\n"
      "  --single <path>         run one image and print instances\n"
      "  --gt-dir <dir>          GT masks; enables accuracy evaluation\n"
      "  --gt-instances <dir>    instance_id,class_id sidecars (ADE20K)\n"
      "  --out-dir <dir>         writes per_image.csv\n"
      "  --resolution <n>        input resolution, default 504 (multiple of 12)\n"
      "  --conf <f>              confidence threshold, default 0.25\n"
      "  --num-classes <n>       number of real classes, default 4\n"
      "  --threads <n>           CPU threads, default 4\n"
      "  --warmup <n>            warmup iterations, default 3\n"
      "  --max-images <n>        limit, 0 = all\n"
      "  --quiet                 suppress progress\n");
}

}  // namespace

int main(int argc, char** argv) {
  rfdetrseg::Config cfg;
  std::string single;
  bool quiet = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "[error] %s is missing a value\n", name);
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
    } else if (a == "--gt-instances") {
      cfg.gt_instances_dir = next("--gt-instances");
    } else if (a == "--out-dir") {
      cfg.out_dir = next("--out-dir");
    } else if (a == "--resolution") {
      cfg.resolution = atoi(next("--resolution").c_str());
    } else if (a == "--conf") {
      cfg.conf = atof(next("--conf").c_str());
    } else if (a == "--num-classes") {
      cfg.num_classes = atoi(next("--num-classes").c_str());
    } else if (a == "--threads") {
      cfg.threads = atoi(next("--threads").c_str());
    } else if (a == "--warmup") {
      cfg.warmup = atoi(next("--warmup").c_str());
    } else if (a == "--max-images") {
      cfg.max_images = atoi(next("--max-images").c_str());
    } else if (a == "--quiet") {
      quiet = true;
    } else {
      fprintf(stderr, "[error] unknown argument: %s\n", a.c_str());
      PrintUsage();
      return 2;
    }
  }

  if (cfg.model_path.empty()) {
    fprintf(stderr, "[error] --model is required\n\n");
    PrintUsage();
    return 2;
  }
  if (!single.empty()) {
    std::string err;
    if (!rfdetrseg::RunOne(cfg, single, &err)) {
      fprintf(stderr, "[error] %s\n", err.c_str());
      return 1;
    }
    return 0;
  }
  if (cfg.image_dir.empty() && cfg.list_file.empty()) {
    fprintf(stderr, "[error] --image-dir or --list-file is required\n\n");
    PrintUsage();
    return 2;
  }

  printf("\n");
  printf("==============================================================\n");
  printf("=== rfdetr-seg-test: RF-DETR-Seg-Nano (C++ / MNN)\n");
  printf("==============================================================\n\n");

  const auto s = rfdetrseg::RunPipeline(cfg, !quiet);
  const double n = s.n_images > 0 ? static_cast<double>(s.n_images) : 1.0;

  printf("\n[summary]\n");
  printf("  images        %d\n", s.n_images);
  printf("  GT / pred     %d / %d\n", s.n_gt, s.n_pred);
  if (s.n_gt > 0) {
    const auto prf = cxxcommon::ComputePrf(s.n_tp, s.n_fp, s.n_fn);
    printf("  TP/FP/FN      %d / %d / %d\n", s.n_tp, s.n_fp, s.n_fn);
    printf("  P/R/F1        %.4f / %.4f / %.4f\n", prf.precision, prf.recall, prf.f1);
  }
  if (s.n_matched) {
    printf("\n[segmentation]  n_matched=%zu\n", s.n_matched);
    printf("  Dice      %.4f\n", s.dice);
    printf("  IoU       %.4f\n", s.iou);
    printf("  HD95      %.2f px\n", s.hd95);
    printf("  ASSD      %.2f px\n", s.assd);
    printf("  BF1       %.4f\n", s.bf1);
    printf("  area_err  %.4f\n", s.area_err);
    if (s.n_cls_ok > 0 || !cfg.gt_instances_dir.empty()) {
      printf("  cls_acc   %.4f\n",
             static_cast<double>(s.n_cls_ok) / std::max<size_t>(1, s.n_matched));
    }
  }
  printf("\n[speed]\n");
  printf("  inference  %7.2f ms/img\n", s.infer_ms / n);
  for (const auto& [k, v] : s.stage_ms) {
    printf("  %-10s %7.2f ms/img\n", k.c_str(), v / n);
  }
  printf("  post       %7.2f ms/img\n", s.post_ms / n);
  printf("  total      %7.2f ms/img  (%.1f FPS)\n", s.total_ms / n,
         1000.0 * n / std::max(1e-9, s.total_ms));
  printf("\n");
  return 0;
}
