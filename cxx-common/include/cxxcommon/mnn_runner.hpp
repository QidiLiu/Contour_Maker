// MNN 推理封装（Express / Expr::Module API，对应 MNN 3.x）。
//
// MNN 3.x 的 Interpreter 低层 API（getInputs/doInference）在公开头文件中已不完整，
// 因此这里使用官方推荐的 Express 接口：
//   Executor::RuntimeManager::createRuntimeManager(ScheduleConfig) -> RuntimeManager
//   Module::load(inputs, outputs, file, runtimeManager, config)
//   module->forward(VARP) -> VARP
//   VARP::readMap() 读输出数据
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "MNN/ErrorCode.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/Executor.hpp"
#include "MNN/expr/Module.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"

namespace cxxcommon {

enum class Backend { kCpu, kOpenCl };

class MnnRunner {
 public:
  MnnRunner() = default;
  ~MnnRunner();

  MnnRunner(const MnnRunner&) = delete;
  MnnRunner& operator=(const MnnRunner&) = delete;

  // 加载模型。backend=kOpenCl 且环境不可用时自动降级到 CPU。
  bool Load(const std::string& model_path, Backend backend, int num_threads,
            Backend* actual = nullptr, std::string* err = nullptr);

  void Unload();

  int NumOutputs() const { return num_outputs_; }
  const std::string& OutputName(int i) const;

  // 前向。input 为 NCHW float32。
  bool Forward(const float* input, const std::vector<int>& input_shape,
               std::string* err = nullptr);

  // 取第 index 个输出。指针指向 MNN 内部存储，下次 Forward 后失效。
  bool Output(int index, const float** data, std::vector<int>* shape) const;

  double LastLatencyMs() const { return last_ms_; }
  const std::string& ModelName() const { return model_path_; }

 private:
  std::shared_ptr<MNN::Express::Executor> executor_;
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> rt_mgr_;
  std::shared_ptr<MNN::Express::Module> module_;
  std::vector<MNN::Express::VARP> outputs_;
  std::vector<std::string> output_names_;
  int num_outputs_ = 0;

  Backend backend_ = Backend::kCpu;
  std::string model_path_;
  double last_ms_ = 0.0;
};

bool OpenClAvailable();
const char* BackendName(Backend b);

}  // namespace cxxcommon