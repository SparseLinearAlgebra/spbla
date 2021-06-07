#include "dcsr.hpp"
#include <cassert>

namespace clbool::dcsr {

    void kronecker_product(Controls &controls,
                           matrix_dcsr &matrix_c,
                           const matrix_dcsr &matrix_a,
                           const matrix_dcsr &matrix_b) {
        uint32_t c_nnz = matrix_a.nnz() * matrix_b.nnz();
        uint32_t c_nzr = matrix_a.nzr() * matrix_b.nzr();
        uint32_t c_nrows = matrix_a.nrows() * matrix_b.nrows();
        uint32_t c_ncols = matrix_a.ncols() * matrix_b.ncols();

        if (c_nnz == 0) {
            matrix_c = matrix_dcsr(c_nrows, c_ncols);
            return;
        }

        cl::Buffer c_rpt; CLB_CREATE_BUF(c_rpt = utils::create_buffer(controls, c_nzr + 1));
        cl::Buffer c_rows; CLB_CREATE_BUF(c_rows =utils::create_buffer(controls, c_nzr));
        cl::Buffer c_cols; CLB_CREATE_BUF(c_cols = utils::create_buffer(controls, c_nnz));

        //  -------------------- form rpt and rows -------------------------------
        auto cnt_nnz = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t>
                ("dcsr_kronecker", "count_nnz_per_row");
        cnt_nnz.set_work_size(c_nzr);

        CLB_RUN(
                TIME_RUN("count nnz in kronecker executed in: ",
                       cnt_nnz.run(controls, c_rpt, c_rows,
                                   matrix_a.rpt_gpu(), matrix_b.rpt_gpu(),
                                   matrix_a.rows_gpu(), matrix_b.rows_gpu(),
                                   c_nzr, matrix_b.nzr(), matrix_b.nrows()).wait()
                ) );

        // c_rpt becomes an array of pointers after exclusive pref sum

        uint32_t total_sum;
        {
            START_TIMING
            prefix_sum(controls, c_rpt, total_sum, c_nzr + 1);
            END_TIMING("prefix_sum finished in: ")
        }
        assert(total_sum == c_nnz);

        // -------------------- form cols -------------------------------

        auto kronecker = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t, uint32_t>
                ("dcsr_kronecker", "calculate_kronecker_product");
        kronecker.set_work_size(c_nnz);

        CLB_RUN(
                TIME_RUN("kronecker indices found in: ",
                       kronecker.run(controls, c_rpt, c_cols, matrix_a.rpt_gpu(), matrix_b.rpt_gpu(),
                                     matrix_a.cols_gpu(), matrix_b.cols_gpu(),
                                     matrix_b.nzr(),
                                     c_nnz, c_nzr, matrix_b.ncols())));

        matrix_c = matrix_dcsr(std::move(c_rpt), std::move(c_rows), std::move(c_cols),
                               c_nrows, c_ncols, c_nnz, c_nzr);

    }
}