cmake_minimum_required(VERSION 3.1)

project(libgpu)

set(CMAKE_CXX_STANDARD 11)

add_executable(hexdumparray libgpu/hexdumparray.cpp)
add_executable(make_source_map libgpu/make_source_map.cpp)

function(convertIntoHeader sourceFile headerFile arrayName)
    add_custom_command(
            OUTPUT ${PROJECT_SOURCE_DIR}/${headerFile}

            COMMAND hexdumparray ${PROJECT_SOURCE_DIR}/${sourceFile} ${PROJECT_SOURCE_DIR}/${headerFile} ${arrayName}

            DEPENDS ${PROJECT_SOURCE_DIR}/${sourceFile} hexdumparray
    )
endfunction()

function(makeSourceMap kernels_list)
    add_custom_command(
            OUTPUT ${PROJECT_SOURCE_DIR}/src/cl/headers/headers_map.hpp

            COMMAND make_source_map ${PROJECT_SOURCE_DIR}/src/cl/headers/headers_map.hpp ${kernels_list}
    )
endfunction()