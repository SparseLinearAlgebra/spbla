#include <algorithm>
#include "../common/cl_includes.hpp"
#include "coo_tests.hpp"
#include "controls.hpp"
#include "../coo/coo_utils.hpp"
#include "../cl/headers/bitonic_sort_new.h"

void clbool::test::testNewBitonicSort() {
    Controls controls = utils::create_controls();
    for (uint32_t i = 10; i < 10000; i += 50) {
//        uint32_t i =  512;
        std::cout << "i = " << i << std::endl;
        cpu_buffer data(i);
        utils::fill_random_buffer(data);
        cl::Buffer d_data(controls.queue, data.begin(), data.end(), false);
        uint32_t block_size = 64;
        auto p = program<cl::Buffer, uint32_t>(bitonic_sort_new_kernel, bitonic_sort_new_kernel_length)
                .set_block_size(block_size)
                .set_needed_work_size(block_size)
                .set_kernel_name("bitonic_sort");
        p.run(controls, d_data, i);
        std::sort(data.begin(), data.end());
        utils::compare_buffers(controls, d_data, data, i, "i = " + std::to_string(i));
    }
}

