cmake_minimum_required(VERSION 3.15)
project(nsparse LANGUAGES CXX CUDA)

add_library(nsparse INTERFACE)
target_include_directories(nsparse INTERFACE include/)
target_link_libraries(nsparse INTERFACE cub)
target_compile_options(nsparse INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --expt-extended-lambda>)

if (SPBLA_BUILD_NSPARSE_TESTS)
    add_subdirectory(test)
endif()
