cmake_minimum_required(VERSION 3.16)
project(det_unet_test CXX)

add_executable(det_unet_test
  src/main.cpp
  src/pipeline.cpp
)

target_include_directories(det_unet_test PRIVATE
  ${CMAKE_SOURCE_DIR}/cxx-common/include
)

target_link_libraries(det_unet_test PRIVATE cxxcommon)