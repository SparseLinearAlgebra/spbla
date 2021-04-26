#pragma once

#include "dcsr.hpp"

namespace clbool {
    void build_groups_and_allocate_hash(Controls &controls,
                                        std::vector<cpu_buffer> &cpu_workload_groups,
                                        cl::Buffer &nnz_estimation,
                                        const matrix_dcsr &a,
                                        cl::Buffer &global_hash_tables,
                                        cl::Buffer &global_hash_tables_offset
    );

    void count_nnz(Controls &controls,
                   const cpu_buffer &groups_length,
                   const cpu_buffer &groups_pointers,

                   const cl::Buffer &gpu_workload_groups,
                   cl::Buffer &nnz_estimation,

                   const matrix_dcsr &a,
                   const matrix_dcsr &b,

                   cl::Buffer &global_hash_tables,
                   const cl::Buffer &global_hash_tables_offset

    );

    void fill_nnz(Controls &controls,
                  const cpu_buffer &groups_length,
                  const cpu_buffer &groups_pointers,

                  const cl::Buffer &gpu_workload_groups,
                  cl::Buffer &pre_matrix_rows_pointers,

                  matrix_dcsr &c,
                  const matrix_dcsr &a,
                  const matrix_dcsr &b,

                  const cl::Buffer &global_hash_tables,
                  cl::Buffer &global_hash_tables_offset
    );

    void matrix_multiplication_hash(Controls &controls,
                                    matrix_dcsr &matrix_out,
                                    const matrix_dcsr &a,
                                    const matrix_dcsr &b);
}