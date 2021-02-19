cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(spbla LANGUAGES CXX)

option(SPBLA_WITH_CPU_BACKEND "Build library with cpu backend as fallback" ON)
option(SPBLA_WITH_CUDA_BACKEND "Build library with cuda backend" OFF)
option(SPBLA_WITH_OPENCL_BACKEND "Build library with opencl backend" OFF)

# Make sure our own modules are found
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

add_subdirectory(spbla)
