#pragma once

#include "../core/controls.hpp"
#include "../core/matrix_coo.hpp"
#include "../core/matrix_dcsr.hpp"
#include "../core/cpu_matrices.hpp"
#include "../common/matrices_conversions.hpp"
#include "../core/kernel.hpp"


namespace clbool::dcsr {

    void submatrix(Controls &controls,
                   matrix_dcsr &matrix_out,
                   const matrix_dcsr &matrix_in,
                   uint32_t i, uint32_t j,
                   uint32_t nrows, uint32_t ncols);

    void transpose(Controls &controls,
                   matrix_dcsr &matrix_out,
                   const matrix_dcsr &matrix_in);

    void reduce(Controls &controls,
                matrix_dcsr &matrix_out,
                const matrix_dcsr &matrix_in);

    void matrix_multiplication(Controls &controls,
                               matrix_dcsr &matrix_out,
                               const matrix_dcsr &a,
                               const matrix_dcsr &b);


    void matrix_multiplication_hash(Controls &controls,
                           matrix_dcsr &matrix_out,
                           const matrix_dcsr &a,
                           const matrix_dcsr &b);

    void kronecker_product(Controls &controls,
                           matrix_dcsr& matrix_c,
                           const matrix_dcsr& matrix_a,
                           const matrix_dcsr& matrix_b);

}