#include "coo_tests.hpp"

#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"

using namespace clbool::coo_utils;

void clbool::test::testCountWorkload() {
    Controls controls = utils::create_controls();
    for (uint32_t i = 120; i < 10000; i += 50) {
        std::cout << "i = " << i << std::endl;
        uint32_t max_size = i;
        uint32_t nnz_limit = max_size * 10;
        matrix_dcsr_cpu a_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_limit, max_size));
//    matrix_dcsr_cpu b_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_limit + 1, max_size));

//    if (nnz_limit < 50) {
//        coo_utils::print_matrix(a_cpu);
//        coo_utils::print_matrix(b_cpu);
//    }

        matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
//    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);

        // get workload from gpu

        cl::Buffer nnz_estimation;
        count_workload(controls, nnz_estimation, a_gpu, a_gpu);

        std::cout << "finish gpu counting" << std::endl;

        // get workload from cpu
        cpu_buffer nnz_estimation_cpu(a_gpu.nzr());
        coo_utils::get_workload(nnz_estimation_cpu, a_cpu, a_cpu);

        std::cout << "finish cpu counting" << std::endl;

        // compare buffers
        utils::compare_buffers(controls, nnz_estimation, nnz_estimation_cpu, a_gpu.nzr());
    }
}

