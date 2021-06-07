

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
        coo_merge.set_work_size(merged_size);

        cl::Buffer merged_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        cl::Buffer merged_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        CLB_RUN(
        TIME_RUN("coo_merge run in: ",
        coo_merge.run(controls, merged_rows, merged_cols, a.rows_gpu(), a.cols_gpu(),
                      b.rows_gpu(), b.cols_gpu(), a.nnz(), b.nnz()).wait()));

        merged_rows_out = std::move(merged_rows);
        merged_cols_out = std::move(merged_cols);
    }

    void matrix_addition(Controls &controls,
                         matrix_coo &matrix_out,
                         const matrix_coo &a,
                         const matrix_coo &b) {
        if (a.nrows() != b.nrows() || a.ncols() != b.ncols()) {
            std::stringstream s;
            s << "Invalid matrixes size! a: " << a.nrows() << " x " << a.ncols() <<
              ", b: " << b.nrows() << " x " << b.ncols();
            CLB_RAISE(s.str(), CLBOOL_INVALID_ARGUMENT);
        }

        if (a.empty() && b.empty()) {
            matrix_out = matrix_coo(a.nrows(), a.ncols());
            return;
        }

        if (a.empty() || b.empty()) {
            const matrix_coo &empty = a.empty() ? a : b;
            const matrix_coo &filled = a.empty() ? b : a;

            if (&matrix_out == &filled) return;

            cl::Buffer rows;
            CLB_CREATE_BUF(rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * filled.nnz()));
            cl::Buffer cols;
            CLB_CREATE_BUF(cols = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * filled.nnz()));

            CLB_CREATE_BUF(controls.queue.enqueueCopyBuffer(filled.rows_gpu(), rows, 0, 0, sizeof(uint32_t) * filled.nnz()));
            CLB_CREATE_BUF(controls.queue.enqueueCopyBuffer(filled.cols_gpu(), cols, 0, 0, sizeof(uint32_t) * filled.nnz()));
            matrix_out = matrix_coo(rows, cols, filled.nrows(), filled.ncols(), filled.nnz());

            return;
        }


        cl::Buffer merged_rows;
        cl::Buffer merged_cols;

        merge(controls, merged_rows, merged_cols, a, b);
        matrix_out = matrix_coo(controls, a.nrows(), a.ncols(), a.nnz() + b.nnz(), merged_rows, merged_cols, true,
                                false);
    }
}