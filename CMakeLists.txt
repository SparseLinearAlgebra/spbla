# CuBool library Cmake config file
# Add this file as sub-directory to your project to use library functionality

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(spbla LANGUAGES CXX)

# Exposed to the user build options
option(SPBLA_WITH_CUDA          "Build library with cuda backend (default)" ON)
option(SPBLA_WITH_OPENCL        "Build library with opencl backend (default)" ON)
option(SPBLA_WITH_SEQUENTIAL    "Build library with cpu sequential backend (fallback)" ON)
option(SPBLA_BUILD_TESTS        "Build project unit-tests with gtest" ON)
option(SPBLA_COPY_TO_PY_PACKAGE "Copy compiled shared library into python package folder (for package use purposes)" ON)
option(SPBLA_WITH_CUB           "Build with bundled cub sources (enable for CUDA SDK version <= 10)" OFF)

set(SPBLA_VERSION_MAJOR 1)
set(SPBLA_VERSION_MINOR 0)
set(SPBLA_VERSION_SUB 0)

set(SPBLA_DEBUG OFF)
set(SPBLA_RELEASE OFF)

if (${CMAKE_BUILD_TYPE} MATCHES Release)
    message(STATUS "Build spbla in release mode")
    set(SPBLA_RELEASE ON)
elseif (${CMAKE_BUILD_TYPE} MATCHES Debug)
    message(STATUS "Build spbla in debug mode")
    set(SPBLA_DEBUG ON)
else()
    message(STATUS "Build spbla in release mode (default: was not specified)")
    set(SPBLA_RELEASE ON)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure cuda dependencies
if (SPBLA_WITH_CUDA)
    if (SPBLA_WITH_CUB)
        message(STATUS "Add cub as cuda utility")
        set(CUB_ENABLE_HEADER_TESTING OFF CACHE BOOL "" FORCE)
        set(CUB_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(CUB_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)

        add_subdirectory(deps/cub)
        add_library(cub INTERFACE IMPORTED)
        target_link_libraries(cub INTERFACE CUB::CUB)
    endif()

    message(STATUS "Add nsparse library as crs matrix multiplication backend")
    add_subdirectory(deps/nsparse)
endif()


if (SPBLA_WITH_OPENCL)
    add_subdirectory(deps/clbool)
endif()

if (SPBLA_BUILD_TESTS)
    message(STATUS "Add googletest as unit-testing library")
    add_subdirectory(deps/gtest)
endif()

# Actual cxx implementation
add_subdirectory(spbla)

# Copy scripts into binary directory
file(COPY scripts DESTINATION ${CMAKE_BINARY_DIR}/)

# Copy python related stuff
file(COPY python DESTINATION ${CMAKE_BINARY_DIR}/)