#pragma once

#include "dcsr.hpp"
#include "kernel.hpp"
#include "controls.hpp"
#include "matrix_coo.hpp"
#include "matrix_dcsr.hpp"
#include "../common/matrices_conversions.hpp"

#define MERGE_ALGORITHM 1
#define HASH_ALGORITHM 2



namespace clbool::dcsr {
    void prepare_positions(Controls &controls,
                           cl::Buffer &positions,
                           const cl::Buffer &array,
                           uint32_t size,
                           const std::string &program_name
    );


    void count_workload(Controls &controls,
                        cl::Buffer &nnz_estimation_out,
                        const matrix_dcsr &a,
                        const matrix_dcsr &b);

    void build_groups_and_allocate_new_matrix(Controls &controls,
                                              matrix_dcsr &pre,
                                              std::vector<cpu_buffer> &cpu_workload_groups,
                                              cl::Buffer &nnz_estimation,
                                              const matrix_dcsr &a,
                                              uint32_t b_cols,

                                              cl::Buffer &aux_pointers,
                                              cl::Buffer &aux_mem
    );

    uint32_t get_group(uint32_t size);


    void run_kernels(Controls &controls,
                     const cpu_buffer &groups_length,
                     const cpu_buffer &groups_pointers,

                     const cl::Buffer &gpu_workload_groups,
                     cl::Buffer &nnz_estimation,

                     const matrix_dcsr &pre,
                     const matrix_dcsr &a,
                     const matrix_dcsr &b,

                     const cl::Buffer &aux_mem_pointers,
                     cl::Buffer &aux_mem
    );

    void write_bins_info(Controls &controls,
                         cl::Buffer &gpu_workload_groups,
                         const std::vector<cpu_buffer> &cpu_workload_groups,
                         cpu_buffer &groups_pointers,
                         cpu_buffer &groups_length
    );

    void create_final_matrix(Controls &controls,
                             matrix_dcsr &c,
                             cl::Buffer &nnz_estimation,
                             const matrix_dcsr &pre,

                             const cl::Buffer &gpu_workload_groups,
                             const cpu_buffer &groups_pointers,
                             const cpu_buffer &groups_length,

                             const matrix_dcsr &a
    );

    void matrix_multiplication(Controls &controls,
                               matrix_dcsr &matrix_out,
                               const matrix_dcsr &a,
                               const matrix_dcsr &b);

    void set_positions(Controls &controls,
                       cl::Buffer &c_rows_pointers,
                       cl::Buffer &c_rows_compressed,
                       const cl::Buffer &nnz_estimation,
                       const cl::Buffer &a_rows_compressed,
                       const cl::Buffer &positions,
                       uint32_t c_nnz,
                       uint32_t old_nzr,
                       uint32_t c_nzr
    );

}