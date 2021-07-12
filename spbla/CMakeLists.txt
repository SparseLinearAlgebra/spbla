project(spbla-cpp LANGUAGES CXX)

if (SPBLA_WITH_CUDA)
    # If Cuda backend is compiled, we must tell cmake, that we will use Cuda
    enable_language(CUDA)
endif()

# Notify user about selected backend options
if (SPBLA_WITH_SEQUENTIAL)
    message(STATUS "Add CPU sequential fallback backend")
endif()
    if (SPBLA_WITH_CUDA)
    message(STATUS "Add CUDA backend for GPGPU computations")
endif()
if (SPBLA_WITH_OPENCL)
    message(STATUS "Add OpenCL backend for GPGPU computations")
endif()

set(TARGET_NAME spbla)
set(TARGET_FILE_NAME)
set(DEFINES_LIST)

# Platform checks
if(APPLE)
    list(APPEND DEFINES_LIST SPBLA_PLATFORM_MACOS)
    set(TARGET_FILE_NAME "lib${TARGET_NAME}.dylib")
elseif(UNIX)
    list(APPEND DEFINES_LIST SPBLA_PLATFORM_LINUX)
    set(TARGET_FILE_NAME "lib${TARGET_NAME}.so")
elseif(WIN32)
    list(APPEND DEFINES_LIST SPBLA_PLATFORM_WIN)
    set(TARGET_FILE_NAME "lib${TARGET_NAME}.dll")
else()
    message(FATAL_ERROR "Platform not specified")
endif()

# Library sources
set(SPBLA_SOURCES
    sources/core/config.hpp
    sources/core/version.hpp
    sources/core/error.hpp
    sources/core/library.cpp
    sources/core/library.hpp
    sources/core/matrix.cpp
    sources/core/matrix.hpp
    sources/io/logger.cpp
    sources/io/logger.hpp
    sources/utils/exclusive_scan.hpp
    sources/utils/timer.hpp
    sources/utils/csr_utils.cpp
    sources/utils/csr_utils.hpp)

set(SPBLA_C_API_SOURCES
    include/spbla/spbla.h
    sources/spbla_GetAbout.cpp
    sources/spbla_GetVersion.cpp
    sources/spbla_GetLicenseInfo.cpp
    sources/spbla_GetDeviceCaps.cpp
    sources/spbla_Initialize.cpp
    sources/spbla_Finalize.cpp
    sources/spbla_SetupLogger.cpp
    sources/spbla_Matrix_New.cpp
    sources/spbla_Matrix_Build.cpp
    sources/spbla_Matrix_SetElement.cpp
    sources/spbla_Matrix_SetMarker.cpp
    sources/spbla_Matrix_Marker.cpp
    sources/spbla_Matrix_ExtractPairs.cpp
    sources/spbla_Matrix_ExtractSubMatrix.cpp
    sources/spbla_Matrix_Duplicate.cpp
    sources/spbla_Matrix_Transpose.cpp
    sources/spbla_Matrix_Nvals.cpp
    sources/spbla_Matrix_Nrows.cpp
    sources/spbla_Matrix_Ncols.cpp
    sources/spbla_Matrix_Free.cpp
    sources/spbla_Matrix_Reduce.cpp
    sources/spbla_Matrix_EWiseAdd.cpp
    sources/spbla_MxM.cpp
    sources/spbla_Kronecker.cpp)

set(SPBLA_BACKEND_SOURCES
    sources/backend/backend_base.hpp
    sources/backend/matrix_base.hpp)

set(SPBLA_CUDA_SOURCES)
set(SPBLA_OPENCL_SOURCES)
set(SPBLA_SEQUENTIAL_SOURCES)

# Cuda backend sources
if (SPBLA_WITH_CUDA)
    set(SPBLA_CUDA_SOURCES
        sources/cuda/cuda_backend.hpp
        sources/cuda/cuda_backend.cu
        sources/cuda/cuda_instance.hpp
        sources/cuda/cuda_instance.cu
        sources/cuda/cuda_instance.cpp
        sources/cuda/cuda_matrix.hpp
        sources/cuda/cuda_matrix.cu
        sources/cuda/cuda_matrix_build.cu
        sources/cuda/cuda_matrix_extract.cu
        sources/cuda/cuda_matrix_ewiseadd.cu
        sources/cuda/cuda_matrix_kronecker.cu
        sources/cuda/cuda_matrix_multiply.cu
        sources/cuda/cuda_matrix_transpose.cu
        sources/cuda/cuda_matrix_reduce.cu
        sources/cuda/cuda_matrix_extract_sub_matrix.cu
        sources/cuda/kernels/slow_sort.cuh
        sources/cuda/kernels/bin_search.cuh
        sources/cuda/kernels/sptranspose.cuh
        sources/cuda/kernels/sptranspose2.cuh
        sources/cuda/kernels/spkron.cuh
        sources/cuda/kernels/spmerge.cuh
        sources/cuda/kernels/spreduce.cuh
        sources/cuda/kernels/spsubmatrix.cuh)
endif()

# OpenCL backend related stuff
if (SPBLA_WITH_OPENCL)
    set(SPBLA_OPENCL_SOURCES
        sources/opencl/opencl_backend.cpp
        sources/opencl/opencl_backend.hpp
        sources/opencl/opencl_matrix.cpp
        sources/opencl/opencl_matrix.hpp
        sources/opencl/opencl_matrix_build.cpp
        sources/opencl/opencl_matrix_extract.cpp
        sources/opencl/opencl_matrix_ewiseadd.cpp
        sources/opencl/opencl_matrix_kronecker.cpp
        sources/opencl/opencl_matrix_multiply.cpp
        sources/opencl/opencl_matrix_transpose.cpp
        sources/opencl/opencl_matrix_reduce.cpp
        sources/opencl/opencl_matrix_extract_sub_matrix.cpp)
endif()

# Cpu sequential backend sources
if (SPBLA_WITH_SEQUENTIAL)
    set(SPBLA_SEQUENTIAL_SOURCES
        sources/sequential/sq_backend.cpp
        sources/sequential/sq_backend.hpp
        sources/sequential/sq_matrix.cpp
        sources/sequential/sq_matrix.hpp
        sources/sequential/sq_csr_data.hpp
        sources/sequential/sq_transpose.cpp
        sources/sequential/sq_transpose.hpp
        sources/sequential/sq_kronecker.cpp
        sources/sequential/sq_kronecker.hpp
        sources/sequential/sq_ewiseadd.cpp
        sources/sequential/sq_ewiseadd.hpp
        sources/sequential/sq_spgemm.cpp
        sources/sequential/sq_spgemm.hpp
        sources/sequential/sq_reduce.cpp
        sources/sequential/sq_reduce.hpp
        sources/sequential/sq_submatrix.cpp
        sources/sequential/sq_submatrix.hpp)
endif()

# Shared library object config
add_library(spbla SHARED
    ${SPBLA_SOURCES}
    ${SPBLA_C_API_SOURCES}
    ${SPBLA_BACKEND_SOURCES}
    ${SPBLA_OPENCL_SOURCES}
    ${SPBLA_CUDA_SOURCES}
    ${SPBLA_SEQUENTIAL_SOURCES})

target_include_directories(spbla PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(spbla PRIVATE ${CMAKE_CURRENT_LIST_DIR}/sources)

target_compile_definitions(spbla PRIVATE SPBLA_EXPORTS)
target_compile_definitions(spbla PRIVATE SPBLA_VERSION_MAJOR=${SPBLA_VERSION_MAJOR})
target_compile_definitions(spbla PRIVATE SPBLA_VERSION_MINOR=${SPBLA_VERSION_MINOR})
target_compile_definitions(spbla PRIVATE SPBLA_VERSION_SUB=${SPBLA_VERSION_SUB})

target_compile_features(spbla PUBLIC cxx_std_14)

target_compile_options(spbla PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wall>)
target_compile_options(spbla PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>: -O2>)
target_compile_options(spbla PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>: -O0>)

set_target_properties(spbla PROPERTIES CXX_STANDARD 17)
set_target_properties(spbla PROPERTIES CXX_STANDARD_REQUIRED ON)

# Platform defines
foreach(DEFINE ${DEFINES_LIST})
    target_compile_definitions(spbla PUBLIC ${DEFINE})
endforeach()

# Cuda specifics
if (SPBLA_WITH_CUDA)
    set_target_properties(spbla PROPERTIES CUDA_STANDARD 14)
    set_target_properties(spbla PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_target_properties(spbla PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_options(spbla PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math -Xptxas -O2>)

    target_compile_definitions(spbla PUBLIC SPBLA_WITH_CUDA)
    target_link_libraries(spbla PRIVATE nsparse)
endif()

# OpenCL specifics
if (SPBLA_WITH_OPENCL)
    # required flags and deps goes here
    target_link_libraries(spbla PRIVATE clbool)
    target_compile_definitions(spbla PUBLIC SPBLA_WITH_OPENCL)
endif()

# Sequential Cpu based backend
if (SPBLA_WITH_SEQUENTIAL)
    target_compile_definitions(spbla PUBLIC SPBLA_WITH_SEQUENTIAL)
endif()

# If tests enabled, add tests sources to the build
if (SPBLA_BUILD_TESTS)
    add_library(testing INTERFACE)
    target_include_directories(testing INTERFACE ${CMAKE_CURRENT_LIST_DIR}/utils/)
    target_link_libraries(testing INTERFACE spbla)
    target_link_libraries(testing INTERFACE gtest)

    message(STATUS "Add unit tests directory to the project")
    add_subdirectory(tests)
endif()

set(LIBRARY_FILE_NAME ${TARGET_FILE_NAME})

# Copy spbla library after build if allowed
if (SPBLA_COPY_TO_PY_PACKAGE)
    add_custom_command(
        TARGET spbla POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E
        copy
        "${CMAKE_BINARY_DIR}/spbla/${LIBRARY_FILE_NAME}"
        "${CMAKE_BINARY_DIR}/python/pyspbla"
        COMMENT "Copy ${LIBRARY_FILE_NAME} compiled lib into python folder")
endif()

# Copy spbla lib into tests folder for linkage on win
if(WIN32 AND SPBLA_BUILD_TESTS)
    add_custom_command(
            TARGET spbla POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E
            copy
            "${CMAKE_BINARY_DIR}/spbla/${LIBRARY_FILE_NAME}"
            "${CMAKE_BINARY_DIR}/spbla/tests"
            COMMENT "Copy ${LIBRARY_FILE_NAME} compiled libs into test directory"
    )
endif()