#include "coo_tests.hpp"


#include "coo_tests.hpp"
#include "../common/cl_includes.hpp"
#include "controls.hpp"
#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"

using namespace clbool::coo_utils;
using namespace clbool::utils;
const uint32_t BINS_NUM = 38;


void clbool::test::testESC() {
    Controls controls = utils::create_controls();
    uint32_t max_size = 400;
    for (int i = 234523; i  < 234523 + 20; ++i) {
        auto generated = generate_random_matrices_esc(max_size + (i % 234523), i);
        matrix_dcsr_cpu a_cpu = generated.first;
        matrix_dcsr_cpu b_cpu = generated.second;
        matrix_dcsr_cpu c_cpu;
//
//    printf("a_cpu: \n");
//    print_matrix(a_cpu);
//    printf("b_cpu: \n");
//    print_matrix(b_cpu);

        matrix_multiplication_cpu(c_cpu, a_cpu, b_cpu);
//    printf("c_cpu: \n");
//    print_matrix(c_cpu);

        matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
        matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
        matrix_dcsr c_gpu;
//    print_matrix(controls, a_gpu);
//    print_matrix(controls, b_gpu);
        matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);
        compare_matrices(controls, c_gpu, c_cpu);
    }
}




