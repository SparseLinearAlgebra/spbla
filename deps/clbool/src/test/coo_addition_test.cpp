#include "coo_tests.hpp"
#include "../common/cl_includes.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../coo/coo_utils.hpp"
#include "../coo/coo_matrix_addition.hpp"


void clbool::test::testMatrixAddition() {
    Controls controls = utils::create_controls();

    for (int i = 100; i < 10000; ++i) {
        for (int j = 100; j < 1000; ++j) {
            matrix_coo_cpu_pairs matrix_res_cpu;
            matrix_coo_cpu_pairs matrix_a_cpu = coo_utils::generate_coo_pairs_cpu(i, 10756);
            matrix_coo_cpu_pairs matrix_b_cpu = coo_utils::generate_coo_pairs_cpu(j, 23341);

            matrix_coo matrix_res_gpu;
            matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
            matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);

            coo_utils::matrix_addition_cpu(matrix_res_cpu, matrix_a_cpu, matrix_b_cpu);

            matrix_addition(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);

            std::vector<uint32_t> rows_cpu;
            std::vector<uint32_t> cols_cpu;

            coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

            utils::compare_buffers(controls, matrix_res_gpu.rows_gpu(), rows_cpu, rows_cpu.size());
            utils::compare_buffers(controls, matrix_res_gpu.cols_gpu(), cols_cpu, cols_cpu.size());
        }
    }

}
