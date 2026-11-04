cmake_minimum_required(VERSION 3.16)
project(rfdetr_seg_test CXX)

add_executable(rfdetr_seg_test
  src/main.cpp
  src/pipeline.cpp
)

target_include_directories(rfdetr_seg_test PRIVATE
  ${CMAKE_SOURCE_DIR}/cxx-common/include
)

target_link_libraries(rfdetr_seg_test PRIVATE cxxcommon)
