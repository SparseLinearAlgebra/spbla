#include "utils.hpp"
#include <algorithm>

namespace clbool::utils {

    void submatrix_cpu(matrix_dcsr_cpu &matrix_out, const matrix_dcsr_cpu &matrix_in,
                       uint32_t i, uint32_t j, uint32_t nrows, uint32_t ncols) {

        cpu_buffer rpt_out;
        cpu_buffer rows_out;
        cpu_buffer cols_out;

        auto rows_begin = std::lower_bound(std::begin(matrix_in.rows()), std::end(matrix_in.rows()), i)
                - std::begin(matrix_in.rows());
        auto rows_end = std::lower_bound(std::begin(matrix_in.rows()), std::end(matrix_in.rows()), i + nrows)
                - std::begin(matrix_in.rows());

        if (rows_begin == rows_end) {
            matrix_out = matrix_dcsr_cpu();
            return;
        }

        LOG << "CPU rows_begin = " << rows_begin << ", rows_end = " << rows_end ;

        uint32_t nnz_count = 0;

        for (auto idx = rows_begin; idx < rows_end; ++idx) {

            auto row_start = matrix_in.cols().begin() + matrix_in.rpt()[idx];
            auto row_end = matrix_in.cols().begin() + matrix_in.rpt()[idx + 1];

            auto subrow_start = std::lower_bound(row_start, row_end, j);
            auto subrow_end = std::lower_bound(row_start, row_end, j + ncols);

            ptrdiff_t subrow_length = subrow_end - subrow_start;
            if (subrow_length == 0) continue;

            for (auto col_idx = subrow_start; col_idx != subrow_end; ++col_idx) {
                cols_out.push_back(*col_idx - j);
            }
            rpt_out.push_back(nnz_count);
            rows_out.push_back(matrix_in.rows()[idx] - i);

            nnz_count += subrow_length;
        }

        rpt_out.push_back(nnz_count);

        LOG << "CPU nnz_count = " << nnz_count;

        matrix_out = matrix_dcsr_cpu(std::move(rpt_out), std::move(rows_out), std::move(cols_out));
    }


    void transpose(matrix_dcsr_cpu &matrix_out, const matrix_dcsr_cpu &matrix_in) {
//        matrix_coo m_coo =
//                &matrix_out == &matrix_in ? dcsr_to_coo_shallow(controls, const_cast<matrix_dcsr &>(matrix_in))
//                                          : dcsr_to_coo_deep(controls, matrix_in);
//        matrix_coo m_coo_tr = matrix_coo(m_coo.ncols(), m_coo.nrows(), m_coo.nnz(), m_coo.cols_gpu(), m_coo.cols_gpu());
//        matrix_out = coo_to_dcsr_shallow(controls, m_coo_tr);
    }

    void reduce(matrix_dcsr_cpu &matrix_out, const matrix_dcsr_cpu &matrix_in) {
        cpu_buffer rows = matrix_in.rows();
        cpu_buffer::size_type n = rows.size();

        cpu_buffer cols(n, 0);
        cpu_buffer rpt(n + 1);

        for (cpu_buffer::size_type i = 0; i <= n; ++i) {
            rpt[i] = i;
        }

        matrix_out = matrix_dcsr_cpu(std::move(rpt), std::move(rows), std::move(cols));
    }
}

