#include <vector>
#include <cstddef>
#include <cstdint>


#include "../common/utils.hpp"
#include "libutils/fast_random.h"
#include "coo_initialization.hpp"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>

#include <program.hpp>
#include "../cl/headers/coo_bitonic_sort.h"

namespace clbool {

    void sort_arrays(Controls &controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n) {

        auto bitonic_begin = program<cl::Buffer, cl::Buffer, uint32_t>
        (coo_bitonic_sort_kernel, coo_bitonic_sort_kernel_length);
        bitonic_begin.set_kernel_name("local_bitonic_begin")
        .set_needed_work_size(utils::round_to_power2(n));

        auto bitonic_global_step = program<cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t>
        (coo_bitonic_sort_kernel, coo_bitonic_sort_kernel_length);
        bitonic_global_step.set_kernel_name("bitonic_global_step")
        .set_needed_work_size(utils::round_to_power2(n));

        auto bitonic_end = program<cl::Buffer, cl::Buffer, uint32_t>
        (coo_bitonic_sort_kernel, coo_bitonic_sort_kernel_length);
        bitonic_end.set_kernel_name("bitonic_local_endings")
        .set_needed_work_size(utils::round_to_power2(n));


        bitonic_begin.run(controls, rows_gpu, cols_gpu, n);

        uint32_t segment_length = controls.block_size * 2;

        while (segment_length < n) {
            segment_length <<= 1;
            bitonic_global_step.run(controls, rows_gpu, cols_gpu, segment_length, 1, n);
            for (unsigned int i = segment_length / 2; i > controls.block_size * 2; i >>= 1) {
                bitonic_global_step.run(controls, rows_gpu, cols_gpu, i, 0, n);
            }
            bitonic_end.run(controls, rows_gpu, cols_gpu, n);

        }

    }
}
