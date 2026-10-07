cmake_minimum_required(VERSION 3.16)
project(two_stage_test CXX)

add_executable(two_stage_test
  src/main.cpp
  src/pipeline.cpp
)

target_include_directories(two_stage_test PRIVATE
  ${CMAKE_SOURCE_DIR}/cxx-common/include
)

target_link_libraries(two_stage_test PRIVATE cxxcommon)