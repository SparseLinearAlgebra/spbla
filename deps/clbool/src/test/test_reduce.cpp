#include "coo_tests.hpp"

#include <random>
#include "../dcsr/dcsr.hpp"
#include <matrices_conversions.hpp>
#include "../coo/coo_utils.hpp"

using namespace clbool::coo_utils;
using namespace clbool::utils;

void clbool::test::test_reduce() {
    Controls controls = create_controls();
    SET_TIMER
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 3000; size < 10000; size += 200) {
            uint32_t max_size = size;
            uint32_t nnz_max = std::max(10u, max_size * k);

            LOG << " ------------------------------- k = " << k << ", size = " << size
                      << " -------------------------------------------\n"
                      << "max_size = " << size << ", nnz_max = " << nnz_max;

            matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_coo_cpu(nnz_max, max_size));

            matrix_dcsr a_gpu;
            {
                START_TIMING
                a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
                END_TIMING("matrix_dcsr_from_cpu: ")
            }


            {
                START_TIMING
                utils::reduce(a_cpu, a_cpu);
                END_TIMING("reduce on CPU: ")
            }


            {
                START_TIMING
                reduce(controls, a_gpu, a_gpu);
                END_TIMING("reduce on DEVICE: ")
            }

            compare_matrices(controls, a_gpu, a_cpu);
        }
    }
}