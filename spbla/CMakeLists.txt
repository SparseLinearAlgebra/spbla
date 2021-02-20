if (SPBLA_WITH_CUDA_BACKEND)
    # If Cuda backend is compiled, we must tell cmake, that we will use Cuda
    project(spbla-cpp LANGUAGES CXX CUDA)
else()
    # Otherwise only C++ with (possibly) OpenCL api will be used
    project(spbla-cpp LANGUAGES CXX)
endif()

# Notify user about selected backend options
if (SPBLA_WITH_CPU_BACKEND)
    message(STATUS "Add SuiteSparse based CPU backend as fallback")
endif()
if (SPBLA_WITH_CUDA_BACKEND)
    message(STATUS "Add cuBool based CUDA backend for GPGPU computations")
endif()
if (SPBLA_WITH_OPENCL_BACKEND)
    message(STATUS "Add clBool based OpenCL backend for GPGPU computations")
endif()

# Source code of the library
set(SPBLA_C_API_SOURCES
    include/spbla/spbla.h
    sources/spbla_Common.hpp
    sources/spbla_Initialize.cpp
    sources/spbla_Finalize.cpp
    sources/spbla_Matrix_New.cpp
    sources/spbla_Matrix_Free.cpp
    sources/spbla_Matrix_Build.cpp
    sources/spbla_Matrix_Extract.cpp
    sources/spbla_Matrix_Nrows.cpp
    sources/spbla_Matrix_Ncols.cpp
    sources/spbla_Matrix_Nvals.cpp
    sources/spbla_Matrix_EWiseAdd.cpp
    sources/spbla_MxM.cpp
    sources/spbla_Kronecker.cpp)

# Generic backend interface
set(SPBLA_BACKEND_SOURCES
    sources/backend/Backend.hpp
    sources/backend/Matrix.hpp)

# Source code for the library core (manages state and validates user input)
set(SPBLA_SOURCES
    sources/core/Defines.hpp
    sources/core/OptionsParser.hpp
    sources/core/Exception.hpp
    sources/core/Library.cpp
    sources/core/Library.hpp
    sources/core/Matrix.cpp
    sources/core/Matrix.hpp)

set(SPBLA_CPU_SOURCES)
set(SPBLA_CUDA_SOURCES)
set(SPBLA_OPENCL_SOURCES)

# Specifics for SuiteSparse based backend
if (SPBLA_WITH_CPU_BACKEND)
    set(SPBLA_CPU_SOURCES
        sources/cpu/Matrix.cpp
        sources/cpu/Matrix.hpp
        sources/cpu/Backend.cpp
        sources/cpu/Backend.hpp)
endif()

# Specifics for Cuda (cuBool) based backend
if (SPBLA_WITH_CUDA_BACKEND)

endif()

# Specifics for OpenCL (clBool) based backend
if (SPBLA_WITH_OPENCL_BACKEND)

endif()

add_library(spbla SHARED
    ${SPBLA_SOURCES}
    ${SPBLA_C_API_SOURCES}
    ${SPBLA_BACKEND_SOURCES}
    ${SPBLA_CPU_SOURCES}
    ${SPBLA_CUDA_SOURCES}
    ${SPBLA_OPENCL_SOURCES})

target_include_directories(spbla PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(spbla PRIVATE ${CMAKE_CURRENT_LIST_DIR}/sources)

# Define while building the library itself
target_compile_definitions(spbla PRIVATE SPBLA_EXPORTS)

target_compile_features(spbla PUBLIC cxx_std_14)
set_target_properties(spbla PROPERTIES CXX_STANDARD 14)
set_target_properties(spbla PROPERTIES CXX_STANDARD_REQUIRED ON)

if (SPBLA_WITH_CPU_BACKEND)
    # Add SuiteSparse dependency (fix link path in future)
    add_library(SuiteSparse INTERFACE)
    target_link_libraries(SuiteSparse INTERFACE graphblas)
    target_link_directories(SuiteSparse INTERFACE /usr/local/lib)

    target_link_libraries(spbla PRIVATE SuiteSparse)
    target_compile_definitions(spbla PRIVATE SPBLA_WITH_CPU_BACKEND)
endif()

# Add unit tests for library cxx sources
if (SPBLA_WITH_TESTS)
    add_library(testing INTERFACE)
    target_include_directories(testing INTERFACE ${CMAKE_CURRENT_LIST_DIR}/utils/)
    target_link_libraries(testing INTERFACE spbla)
    target_link_libraries(testing INTERFACE gtest)

    message(STATUS "Add spbla-cpp unit tests")
    add_subdirectory(tests)
endif()