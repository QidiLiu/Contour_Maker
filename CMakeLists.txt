cmake_minimum_required(VERSION 3.16)
project(ContourMaker CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# ---------------------------------------------------------------- MNN
# MNN 作为 third_party 源码内联编译，使四个子项目都无需预装 MNN。
set(MNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MNN)
if(NOT EXISTS ${MNN_DIR})
  message(FATAL_ERROR
    "未找到 MNN 源码: ${MNN_DIR}\n"
    "请先执行: git clone --depth 1 https://github.com/alibaba/MNN.git third_party/MNN")
endif()

# 直接把 MNN 的核心库与 OpenCL 静态库加入构建树
add_subdirectory(${MNN_DIR} ${CMAKE_BINARY_DIR}/mnn EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------- OpenCL
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
  message(STATUS "OpenCL headers: ${OpenCL_INCLUDE_DIRS}")
else()
  message(STATUS "OpenCL 未找到（不影响 CPU 后端；OpenCL 后端需要 OpenCL 头文件）")
endif()

# ---------------------------------------------------------------- options
option(CM_BUILD_TESTS "构建 two-stage-test / one-stage-test" ON)

# ---------------------------------------------------------------- cxx-common
# 两个 C++ 测试子项目共用的静态库：图像 IO、几何、MNN 封装。
add_library(cxxcommon STATIC
  cxx-common/src/image.cpp
  cxx-common/src/png.cpp
  cxx-common/src/metrics.cpp
  cxx-common/src/postprocess.cpp
  cxx-common/src/mnn_runner.cpp
)
target_include_directories(cxxcommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cxx-common/include
  ${MNN_DIR}/include
  ${MNN_DIR}/include/MNN
  ${MNN_DIR}/express
)
target_compile_definitions(cxxcommon PUBLIC
  MNN_OPENCL_ENABLED=$<BOOL:${OpenCL_FOUND}>
)
if(OpenCL_FOUND)
  target_include_directories(cxxcommon PUBLIC ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(cxxcommon PUBLIC ${OpenCL_LIBRARIES} MNN_CL)
endif()
# MNN_Express 提供 Express::Executor / Expr::Module / VARP 等接口
target_link_libraries(cxxcommon PUBLIC MNN_Express MNN)

# ---------------------------------------------------------------- subprojects
if(CM_BUILD_TESTS)
  add_subdirectory(one-stage-test)
  add_subdirectory(two-stage-test)
endif()

message(STATUS "")
message(STATUS "ContourMaker C++ 构建")
message(STATUS "  OpenCL headers : ${OpenCL_FOUND}")
message(STATUS "  build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "")