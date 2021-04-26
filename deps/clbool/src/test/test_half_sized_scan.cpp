#include "coo_tests.hpp"
#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"
#include "../cl/headers/half_sized_scan.h"

using namespace clbool::coo_utils;
using namespace clbool::utils;
const uint32_t BINS_NUM = 38;

void clbool::test::testScan() {
    Controls controls = create_controls();
    cpu_buffer array(50, 1);
    cl::Buffer array_gpu(controls.context, CL_MEM_READ_WRITE,  array.size() * sizeof(cpu_buffer::value_type));
    controls.queue.enqueueWriteBuffer(array_gpu, CL_TRUE, 0, array.size()
    * sizeof(cpu_buffer::value_type), array.data());
    utils::print_gpu_buffer(controls, array_gpu, array.size());
    try {
        auto half_sized_scan = program<cl::Buffer, uint32_t>(half_sized_scan_kernel, half_sized_scan_kernel_length)
                .set_block_size(32)
                .set_needed_work_size(array.size())
                .set_kernel_name("scan_blelloch_half");

        half_sized_scan.run(controls, array_gpu, array.size());
        utils::print_gpu_buffer(controls, array_gpu, array.size());

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        throw std::runtime_error(exception.str());
    }
}




