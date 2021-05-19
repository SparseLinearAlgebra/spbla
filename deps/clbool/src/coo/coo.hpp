#pragma once

#include <controls.hpp>
#include <matrix_coo.hpp>

namespace clbool::coo {

    void matrix_addition(
            Controls &controls,
            matrix_coo &matrix_out,
            const matrix_coo &a,
            const matrix_coo &b
    );

    void kronecker_product(Controls &controls,
                           matrix_coo &matrix_out,
                           const matrix_coo &matrix_a,
                           const matrix_coo &matrix_b
    );

}