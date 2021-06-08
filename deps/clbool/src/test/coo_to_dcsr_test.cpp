#include <algorithm>
#include "coo_tests.hpp"
#include "../coo/coo_utils.hpp"
#include "../dcsr/dcsr_matrix_multiplication.hpp"


void clbool::test::testCOOtoDCSR() {
    Controls controls = utils::create_controls();
    // ----------------------------------------- create matrices ----------------------------------------

    uint32_t size = 1000463;
    uint32_t max_size = 100024;
    matrix_coo_cpu_pairs m_cpu = coo_utils::generate_coo_pairs_cpu(size, max_size);
    cpu_buffer rows_pointers_cpu;
    cpu_buffer rows_compressed_cpu;
    coo_utils::get_rows_pointers_and_compressed(rows_pointers_cpu, rows_compressed_cpu, m_cpu);

    matrix_coo matrix_gpu = coo_utils::matrix_coo_from_cpu(controls, m_cpu);
    cpu_buffer coo_rows_indices(matrix_gpu.nnz());
    cpu_buffer coo_cols_indices(matrix_gpu.nnz());

    controls.queue.enqueueReadBuffer(matrix_gpu.rows_gpu(), true, 0,
                                      sizeof(matrix_coo::index_type) * matrix_gpu.nnz(), coo_rows_indices.data());
    controls.queue.enqueueReadBuffer(matrix_gpu.cols_gpu(), true, 0,
                                      sizeof(matrix_coo::index_type) * matrix_gpu.nnz(), coo_cols_indices.data());
    utils::print_cpu_buffer(coo_cols_indices, 10);

    matrix_dcsr m_dcsr = coo_to_dcsr_gpu_shallow(controls, matrix_gpu);

    utils::compare_buffers(controls, m_dcsr.rpt_gpu(), rows_pointers_cpu, rows_pointers_cpu.size());
    utils::compare_buffers(controls, m_dcsr.rows_gpu(), rows_compressed_cpu, rows_compressed_cpu.size());


    matrix_coo another_one = dcsr_to_coo_shallow(controls, m_dcsr);
    utils::print_gpu_buffer(controls, another_one.cols_gpu(), 10);
    utils::print_gpu_buffer(controls, another_one.rows_gpu(), 10);
    utils::compare_buffers(controls, another_one.rows_gpu(), coo_rows_indices, coo_rows_indices.size());
    utils::compare_buffers(controls, another_one.cols_gpu(), coo_cols_indices, coo_cols_indices.size());

}