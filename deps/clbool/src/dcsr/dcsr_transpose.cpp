#include "dcsr.hpp"


namespace clbool::dcsr {
    void transpose(Controls &controls, matrix_dcsr &matrix_out, const matrix_dcsr &matrix_in) {
        if (matrix_in.empty()) {
            matrix_out = matrix_dcsr(matrix_in.ncols(), matrix_in.nrows());
            return;
        }
        matrix_coo m_coo = &matrix_out == &matrix_in ? dcsr_to_coo_shallow(controls, const_cast<matrix_dcsr&>(matrix_in))
                : dcsr_to_coo_deep(controls, matrix_in);

        matrix_coo m_coo_tr = matrix_coo(controls, m_coo.ncols(), m_coo.nrows(), m_coo.nnz(),
                                         m_coo.cols_gpu(), m_coo.rows_gpu(),
                                         false, true);
        matrix_out = coo_to_dcsr_shallow(controls, m_coo_tr);
    }
}