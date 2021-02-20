cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(spbla LANGUAGES CXX)

option(SPBLA_WITH_CPU_BACKEND "Build library with cpu backend as fallback" ON)
option(SPBLA_WITH_CUDA_BACKEND "Build library with cuda backend" OFF)
option(SPBLA_WITH_OPENCL_BACKEND "Build library with opencl backend" OFF)

option(SPBLA_WITH_TESTS "Build library source code with provided unit tests" ON)

if (SPBLA_WITH_TESTS)
    message(STATUS "Add gtest dependency for library unit tests")
    add_subdirectory(deps/gtest)
endif()

add_subdirectory(spbla)
