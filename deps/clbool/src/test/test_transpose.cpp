#include "coo_tests.hpp"

#include <random>
#include "../dcsr/dcsr.hpp"
#include <matrices_conversions.hpp>
#include "../coo/coo_utils.hpp"

using namespace clbool::coo_utils;
using namespace clbool::utils;

void clbool::test::test_transpose() {
    Controls controls = create_controls();
    SET_TIMER
    for (uint32_t k = 10; k < 30; ++k) {
        for (int size = 20; size < 400; size += 200) {
            uint32_t max_size = size;
            uint32_t nnz_max = std::max(10u, max_size * k);

            LOG << " ------------------------------- k = " << k << ", size = " << size
                      << " -------------------------------------------\n"
                      << "max_size = " << size << ", nnz_max = " << nnz_max;

            matrix_coo_cpu a_coo_cpu = generate_coo_cpu(nnz_max, max_size);
            matrix_dcsr_cpu a_dcsr_cpu = coo_to_dcsr_cpu(a_coo_cpu);
            a_coo_cpu.transpose();
            matrix_dcsr_cpu a_dcsr_cpu_tr = coo_to_dcsr_cpu(a_coo_cpu);

            matrix_dcsr a_gpu;
            {
                START_TIMING
                a_gpu = matrix_dcsr_from_cpu(controls, a_dcsr_cpu, max_size);
                END_TIMING("matrix_dcsr_from_cpu: ")
            }


            {
                START_TIMING
                transpose(controls, a_gpu, a_gpu);
                END_TIMING("transpose: ")
            }

            compare_matrices(controls, a_gpu, a_dcsr_cpu_tr);
        }
    }
}