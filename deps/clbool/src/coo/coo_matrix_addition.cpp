
#include "controls.hpp"
#include "kernel.hpp"
#include "../common/utils.hpp"
#include "../common/cl_operations.hpp"
#include "coo_utils.hpp"


namespace clbool::coo {

        void merge(Controls &controls,
               cl::Buffer &merged_rows_out,
               cl::Buffer &merged_cols_out,
               const matrix_coo &a,
               const matrix_coo &b) {

        uint32_t merged_size = a.nnz() + b.nnz();

        auto coo_merge = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                        uint32_t, uint32_t>
                        ("merge_path", "merge");
        coo_merge.set_needed_work_size(merged_size);

        cl::Buffer merged_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        cl::Buffer merged_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

        coo_merge.run(controls,
                      merged_rows, merged_cols,
                      a.rows_gpu(), a.cols_gpu(),
                      b.rows_gpu(), b.cols_gpu(),
                      a.nnz(), b.nnz()).wait();

        merged_rows_out = std::move(merged_rows);
        merged_cols_out = std::move(merged_cols);
    }

    void matrix_addition(Controls &controls,
                         matrix_coo &matrix_out,
                         const matrix_coo &a,
                         const matrix_coo &b) {

        cl::Buffer merged_rows;
        cl::Buffer merged_cols;

        merge(controls, merged_rows, merged_cols, a, b);
        matrix_out = matrix_coo(controls, a.nrows(), a.ncols(), a.nnz() + b.nnz(), merged_rows, merged_cols, true, false);
    }
}