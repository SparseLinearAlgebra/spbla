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
    sources/utils/csr_utils.hpp
        )

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
set(SPBLA_SEQUENTIAL_SOURCES)

# Cuda backend sources
if (SPBLA_WITH_CUDA)
    set(SPBLA_CUDA_SOURCES
        sources/cuda/cuda_backend.hpp
        sources/cuda/cuda_backend.cu
        sources/cuda/instance.hpp
        sources/cuda/instance.cu
        sources/cuda/instance.cpp
        sources/cuda/matrix_csr.hpp
        sources/cuda/matrix_csr.cu
        sources/cuda/matrix_csr_build.cu
        sources/cuda/matrix_csr_extract.cu
        sources/cuda/matrix_csr_ewiseadd.cu
        sources/cuda/matrix_csr_kronecker.cu
        sources/cuda/matrix_csr_multiply.cu
        sources/cuda/matrix_csr_transpose.cu
        sources/cuda/matrix_csr_reduce.cu
        sources/cuda/matrix_csr_extract_sub_matrix.cu
        sources/cuda/kernels/slow_sort.cuh
        sources/cuda/kernels/bin_search.cuh
        sources/cuda/kernels/sptranspose.cuh
        sources/cuda/kernels/sptranspose2.cuh
        sources/cuda/kernels/spkron.cuh
        sources/cuda/kernels/spmerge.cuh
        sources/cuda/kernels/spreduce.cuh
        sources/cuda/kernels/spsubmatrix.cuh)
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

# Cuda specifics
if (SPBLA_WITH_CUDA)
    set_target_properties(spbla PROPERTIES CUDA_STANDARD 14)
    set_target_properties(spbla PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_target_properties(spbla PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Settings: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    #target_compile_options(spbla PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    #    # todo: fix this flag later -arch=sm_30 ?
    #    # todo: can we omit arch flag?
    #    -gencode=arch=compute_30,code=sm_30
    #    -gencode=arch=compute_50,code=sm_50
    #    -gencode=arch=compute_52,code=sm_52
    #    -gencode=arch=compute_60,code=sm_60
    #    -gencode=arch=compute_61,code=sm_61
    #    -gencode=arch=compute_61,code=compute_61>)

    target_compile_options(spbla PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math -Xptxas -O2>)

    target_compile_definitions(spbla PRIVATE SPBLA_WITH_CUDA)
    target_link_libraries(spbla PRIVATE nsparse_um)
endif()

# Sequential Cpu based backend
if (SPBLA_WITH_SEQUENTIAL)
    target_compile_definitions(spbla PRIVATE SPBLA_WITH_SEQUENTIAL)
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

# Copy spbla library after build if allowed
if (SPBLA_COPY_TO_PY_PACKAGE)
    # todo: Different platforms has specific naming conventions
    set(LIBRARY_FILE_NAME "libspbla.so")

    add_custom_command(
        TARGET spbla POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E
        copy
        "${CMAKE_BINARY_DIR}/spbla/${LIBRARY_FILE_NAME}"
        "${CMAKE_BINARY_DIR}/python/pyspbla"
        COMMENT "Copy spbla compiled lib into python folder")
endif()