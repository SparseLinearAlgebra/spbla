#pragma once

#include <controls.hpp>
#include <matrix_coo.hpp>
#include <matrix_dcsr.hpp>
#include <matrices_conversions.hpp>

namespace clbool {

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

}