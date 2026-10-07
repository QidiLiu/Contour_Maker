// MNN 推理封装实现（Express / Expr::Module API）。
#include "cxxcommon/mnn_runner.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

namespace cxxcommon {
namespace {
const std::string kEmpty;
}  // namespace

bool OpenClAvailable() {
#ifdef MNN_OPENCL_ENABLED
  return true;
#else
  return false;
#endif
}

const char* BackendName(Backend b) {
  return b == Backend::kOpenCl ? "OpenCL" : "CPU";
}

MnnRunner::~MnnRunner() {
  Unload();
}

const std::string& MnnRunner::OutputName(int i) const {
  if (i < 0 || i >= static_cast<int>(output_names_.size())) return kEmpty;
  return output_names_[i];
}

bool MnnRunner::Load(const std::string& model_path, Backend backend, int num_threads,
                     Backend* actual, std::string* err) {
  Unload();
  model_path_ = model_path;

  Backend use = backend;
  if (use == Backend::kOpenCl && !OpenClAvailable()) {
    fprintf(stderr,
            "[warn] 请求 OpenCL 但当前构建未启用 MNN_OPENCL，自动降级为 CPU。\n");
    use = Backend::kCpu;
  }

  // ---- RuntimeManager
  MNN::ScheduleConfig sched;
  sched.type = (use == Backend::kOpenCl) ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU;
  sched.numThread = (use == Backend::kCpu && num_threads > 0) ? num_threads : 1;
  sched.backupType = MNN_FORWARD_CPU;

  rt_mgr_.reset(MNN::Express::Executor::RuntimeManager::createRuntimeManager(sched));
  if (!rt_mgr_) {
    // OpenCL 后端不可用时回退 CPU 再试
    if (use == Backend::kOpenCl) {
      fprintf(stderr, "[warn] OpenCL RuntimeManager 创建失败，回退 CPU。\n");
      sched.type = MNN_FORWARD_CPU;
      sched.numThread = num_threads > 0 ? num_threads : 1;
      rt_mgr_.reset(MNN::Express::Executor::RuntimeManager::createRuntimeManager(sched));
      use = Backend::kCpu;
    }
  }
  if (!rt_mgr_) {
    if (err) *err = "创建 RuntimeManager 失败";
    return false;
  }

  // ---- 加载模块（不指定输入输出名，由 MNN 使用模型内的默认名）
  MNN::Express::Module::Config mod_cfg;
  mod_cfg.dynamic = false;        // 固定 shape（MNNConvert 导出时已固定尺寸）
  mod_cfg.shapeMutable = false;
  mod_cfg.rearrange = false;

  MNN::Express::Module* m =
      MNN::Express::Module::load({}, {}, model_path.c_str(), rt_mgr_, &mod_cfg);
  if (!m) {
    if (err) *err = "加载模型失败: " + model_path;
    Unload();
    return false;
  }
  module_.reset(m);

  const MNN::Express::Module::Info* info = module_->getInfo();
  if (!info || info->inputs.empty()) {
    if (err) *err = "模型无输入张量";
    Unload();
    return false;
  }
  output_names_ = info->outputNames;
  num_outputs_ = static_cast<int>(output_names_.size());
  if (num_outputs_ == 0) {
    if (err) *err = "模型无输出张量";
    Unload();
    return false;
  }

  backend_ = use;
  if (actual) *actual = use;
  return true;
}

void MnnRunner::Unload() {
  outputs_.clear();
  module_.reset();
  rt_mgr_.reset();
  executor_.reset();
  output_names_.clear();
  num_outputs_ = 0;
  last_ms_ = 0.0;
}

bool MnnRunner::Forward(const float* input, const std::vector<int>& input_shape,
                        std::string* err) {
  if (!module_) {
    if (err) *err = "模型未加载";
    return false;
  }
  if (input_shape.size() != 4) {
    if (err) *err = "仅支持 4 维输入 (N,C,H,W)";
    return false;
  }

  const auto t0 = std::chrono::steady_clock::now();

  MNN::Express::VARP in = MNN::Express::_Input(
      {input_shape[0], input_shape[1], input_shape[2], input_shape[3]},
      MNN::Express::NCHW);
  const size_t n = static_cast<size_t>(input_shape[0]) * input_shape[1] *
                   input_shape[2] * input_shape[3];
  float* dst = in->writeMap<float>();
  if (!dst) {
    if (err) *err = "取输入写指针失败";
    return false;
  }
  std::memcpy(dst, input, n * sizeof(float));

  outputs_ = module_->onForward({in});
  if (outputs_.empty()) {
    if (err) *err = "forward 无输出";
    return false;
  }
  // 固定输出为 INPUT 模式，保证下面的 readMap 从 host 内存读取结果
  for (auto& o : outputs_) o.fix(MNN::Express::VARP::INPUT);

  const auto t1 = std::chrono::steady_clock::now();
  last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return true;
}

bool MnnRunner::Output(int index, const float** data, std::vector<int>* shape) const {
  if (index < 0 || index >= static_cast<int>(outputs_.size())) return false;
  const auto& v = outputs_[index];
  if (!v.get()) return false;
  if (shape) {
    const auto* info = v->getInfo();
    if (!info) return false;
    *shape = info->dim;
  }
  if (data) {
    const float* map = v->readMap<float>();
    if (!map) return false;
    *data = map;
  }
  return true;
}

}  // namespace cxxcommon