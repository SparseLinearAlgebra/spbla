#pragma once

#include "../core/controls.hpp"
#include "../core/kernel.hpp"
#include "../core/matrix_csr.hpp"
#include "../core/cpu_matrices.hpp"
#include "../common/cl_operations.hpp"
#include "../common/utils.hpp"


namespace clbool::csr {
    void matrix_addition(Controls &controls, matrix_csr &c, const matrix_csr &a, const matrix_csr &b);

}