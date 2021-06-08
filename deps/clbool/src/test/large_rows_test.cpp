#include "coo_tests.hpp"


#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"

using namespace clbool::coo_utils;
using namespace clbool::utils;



const uint32_t BINS_NUM = 38;

void clbool::test::largeRowsTest() {
    Controls controls = utils::create_controls();
    uint32_t max_size = 1000;
    auto generated = generate_random_matrices_large(max_size, 342787282);

    matrix_dcsr_cpu a_cpu = generated.first;
    matrix_dcsr_cpu b_cpu = generated.second;
    matrix_dcsr_cpu c_cpu;

    matrix_multiplication_cpu(c_cpu, a_cpu, b_cpu);
    std::cout << "matrix_multiplication_cpu finished" << std::endl;
//    print_matrix(c_cpu);

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
    matrix_dcsr c_gpu;

    matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);

//    print_matrix(controls, c_gpu);

    compare_matrices(controls, c_gpu, c_cpu);

//    print_matrix(c_cpu);

}

