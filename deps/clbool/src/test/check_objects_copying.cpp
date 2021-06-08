#include "coo_tests.hpp"


#include "coo_tests.hpp"
#include "../common/cl_includes.hpp"
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_dcsr.hpp"
#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"



const uint32_t BINS_NUM = 38;
using namespace clbool::utils;
using namespace clbool::coo_utils;


void clbool::test::checkCopying() {
    Controls controls = utils::create_controls();

    uint32_t nnz_limit = 25;
    uint32_t max_size = 10;

    matrix_coo_cpu_pairs matrix_a_coo_cpu = generate_coo_pairs_cpu(nnz_limit, max_size);
    matrix_dcsr_cpu matrix_a_cpu = coo_pairs_to_dcsr_cpu(matrix_a_coo_cpu);

    std::cout << "matrix_a_coo_cpu: \n";
    print_matrix(matrix_a_coo_cpu);

    std::cout << "matrix_a_cpu: \n";
    print_matrix(matrix_a_cpu);

    matrix_coo a_coo = coo_utils::matrix_coo_from_cpu(controls, matrix_a_coo_cpu);

    matrix_dcsr a_dcsr = coo_to_dcsr_gpu_shallow(controls, a_coo);

    uint32_t value = 239;
    controls.queue.enqueueWriteBuffer(a_coo.cols_gpu(), CL_TRUE, 0, sizeof(uint32_t) * 1, &value);
    print_gpu_buffer(controls, a_dcsr.cols_gpu(), 3);

}

void clbool::test::simpleTestCopy() {
    Controls controls = create_controls();

    cpu_buffer a_cpu(5, 1);
    cpu_buffer b_cpu(2, 2);

    cl::Buffer a_gpu(controls.context, CL_TRUE, sizeof(uint32_t) * a_cpu.size());
    cl::Buffer a_gpu_copy(controls.context, CL_TRUE, sizeof(uint32_t) * a_cpu.size());

    controls.queue.enqueueWriteBuffer(a_gpu, CL_TRUE, 0, sizeof(uint32_t) * a_cpu.size(), a_cpu.data());
    print_gpu_buffer(controls, a_gpu_copy, a_cpu.size());
    controls.queue.enqueueCopyBuffer(a_gpu, a_gpu_copy, 0, 0, sizeof(uint32_t) * a_cpu.size());
    controls.queue.enqueueWriteBuffer(a_gpu, CL_TRUE, 0, sizeof(uint32_t) * b_cpu.size(), b_cpu.data());
    print_gpu_buffer(controls, a_gpu, a_cpu.size());
    print_gpu_buffer(controls, a_gpu_copy, a_cpu.size());
}

