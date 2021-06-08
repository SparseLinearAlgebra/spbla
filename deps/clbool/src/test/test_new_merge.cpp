#include <algorithm>
#include "coo_tests.hpp"

#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"
#include "../cl/headers/new_merge.h"
using namespace clbool::coo_utils;
using namespace clbool::utils;
const uint32_t BINS_NUM = 38;

void clbool::test::test_new_merge_full() {
    Controls controls = create_controls();

    for (int test_size_a = 256; test_size_a < 30000; test_size_a += 500) {
        for (int test_size_b = 256; test_size_b < 30000; test_size_b += 500) {



            cpu_buffer a_cpu(test_size_a);
            cpu_buffer b_cpu(test_size_b);
            cpu_buffer c_cpu;

            fill_random_buffer(a_cpu);
            fill_random_buffer(b_cpu);

            std::sort(a_cpu.begin(), a_cpu.end());
            std::sort(b_cpu.begin(), b_cpu.end());

            std::merge(a_cpu.begin(), a_cpu.end(), b_cpu.begin(), b_cpu.end(),
                       std::back_inserter(c_cpu));

            cl::Buffer a_gpu = cl::Buffer(controls.queue, a_cpu.begin(), a_cpu.end(), false);
            cl::Buffer b_gpu = cl::Buffer(controls.queue, b_cpu.begin(), b_cpu.end(), false);
            cl::Buffer c_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_cpu.size());
//            print_cpu_buffer(c_cpu);

            auto new_merge = program<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>
                    (new_merge_kernel, new_merge_kernel_length)
                    .set_kernel_name("new_merge_full")
                    .set_needed_work_size(a_cpu.size() + b_cpu.size());
            new_merge.run(controls,
                        a_gpu, b_gpu, c_gpu, a_cpu.size(), b_cpu.size());

//            std::cout << "~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~\n";
//            print_gpu_buffer(controls, c_gpu, c_cpu.size());

//            std::cout << c_cpu[349] << std::endl;
            compare_buffers(controls, c_gpu, c_cpu, c_cpu.size());
        }
    }
}