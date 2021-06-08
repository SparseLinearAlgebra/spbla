#include "coo_initialization.hpp"
#include <cstdint>

namespace clbool::coo {

    void sort_arrays(Controls &controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n) {

        auto bitonic_begin = kernel<cl::Buffer, cl::Buffer, uint32_t>
        ("coo_bitonic_sort", "local_bitonic_begin");
        bitonic_begin.set_work_size(utils::round_to_power2(n));

        auto bitonic_global_step = kernel<cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t>
        ("coo_bitonic_sort", "bitonic_global_step");
        bitonic_global_step.set_work_size(utils::round_to_power2(n));

        auto bitonic_end = kernel<cl::Buffer, cl::Buffer, uint32_t>
        ("coo_bitonic_sort", "bitonic_local_endings");
        bitonic_end.set_work_size(utils::round_to_power2(n));

        CLB_RUN(bitonic_begin.run(controls, rows_gpu, cols_gpu, n));

        uint32_t segment_length = controls.block_size * 2;
        cl::Event last;
        while (segment_length < n) {
            segment_length <<= 1;
            CLB_RUN(bitonic_global_step.run(controls, rows_gpu, cols_gpu, segment_length, 1, n));
            for (unsigned int i = segment_length / 2; i > controls.block_size * 2; i >>= 1) {
                CLB_RUN(bitonic_global_step.run(controls, rows_gpu, cols_gpu, i, 0, n));
            }
            CLB_RUN(bitonic_end.run(controls, rows_gpu, cols_gpu, n));
        }
    }
}
