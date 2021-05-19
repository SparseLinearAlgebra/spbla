#pragma once
#include "coo.hpp"

#include "matrix_coo.hpp"

namespace clbool::coo {
    void kronecker_product(Controls &controls,
                           matrix_coo &matrix_out,
                           const matrix_coo &matrix_a,
                           const matrix_coo &matrix_b
    );


}
