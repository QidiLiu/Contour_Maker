cmake_minimum_required(VERSION 3.16)
project(yolo_seg_test CXX)

add_executable(yolo_seg_test
  src/main.cpp
  src/pipeline.cpp
)

target_include_directories(yolo_seg_test PRIVATE
  ${CMAKE_SOURCE_DIR}/cxx-common/include
)

target_link_libraries(yolo_seg_test PRIVATE cxxcommon)