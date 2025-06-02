cmake_minimum_required(VERSION 3.25)
project(tiny_exp LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

add_library(tiny_exp SHARED cpp/tiny_exp.cc)
target_link_libraries(tiny_exp PUBLIC cuda)

add_executable(test_tiny_exp cpp/test_tiny_exp.cc)
target_link_libraries(test_tiny_exp PUBLIC tiny_exp)