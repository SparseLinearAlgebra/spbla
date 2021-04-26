#pragma once

#include "../library_classes/matrix_coo.hpp"

namespace clbool {
    void kronecker_product(Controls &controls,
                           matrix_coo &matrix_out,
                           const matrix_coo &matrix_a,
                           const matrix_coo &matrix_b
    );
}
