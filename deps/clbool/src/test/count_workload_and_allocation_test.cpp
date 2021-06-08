#include "coo_tests.hpp"

#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"

using namespace clbool::coo_utils;
using namespace clbool::utils;
const uint32_t BINS_NUM = 38;

void clbool::test::testCountWorkloadAndAllocation() {
    Controls controls = create_controls();

    uint32_t nnz_limit = 25;
    uint32_t max_size = 10;
    matrix_dcsr_cpu a_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_limit, max_size));
    matrix_dcsr_cpu b_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_limit + 1, max_size));

    if (nnz_limit < 50) {
        coo_utils::print_matrix(a_cpu);
        coo_utils::print_matrix(b_cpu);
    }

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);

    // get workload from gpu

    cl::Buffer nnz_estimation;
    count_workload(controls, nnz_estimation, a_gpu, b_gpu);

    std::cout << "nnz_estimation:\n";
    print_gpu_buffer(controls, nnz_estimation, a_gpu.nzr());
    /* --------------------------------------------------------------------------------------------------------
     *
     *
     *
     *
     *
     *
     * --------------------------------------------------------------------------------------------------------
     */

    std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM);

    cl::Buffer a;
    cl::Buffer b;

    matrix_dcsr pre;
    build_groups_and_allocate_new_matrix(controls,
                                         pre,
                                         cpu_workload_groups, nnz_estimation, a_gpu, b_gpu.ncols(), a, b);

    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_gpu.nzr());
    unsigned int offset = 0;

    for (const auto &group: cpu_workload_groups) {
        if (group.empty()) continue;
        controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, sizeof(uint32_t) * offset,
                                          sizeof(uint32_t) * group.size(), group.data());
        offset += group.size();
    }

    std::cout << "cpu vectors: \n";
    uint32_t group_num = -1;
    for (auto const& item: cpu_workload_groups) {
        ++group_num;
        if (item.empty()) continue;
        std::cout << "group " << group_num << ": ";
        utils::print_cpu_buffer(item);
    }

    std::cout << "gpu workload: \n";
    utils::print_gpu_buffer(controls, gpu_workload_groups, offset);

    std::cout << "pre_rows_pointers: \n";
    utils::print_gpu_buffer(controls, pre.rpt_gpu(), a_gpu.nzr() + 1);
}

