#pragma once

#include "controls.hpp"
#include "matrix_coo.hpp"
#include "matrix_dcsr.hpp"
#include "kernel.hpp"
#include "cl_operations.hpp"
#include "utils.hpp"

namespace clbool {
    matrix_coo dcsr_to_coo_shallow(Controls &controls, matrix_dcsr &a);
    matrix_coo dcsr_to_coo_deep(Controls &controls, const matrix_dcsr &a);
    matrix_dcsr coo_to_dcsr_shallow(Controls &controls, const matrix_coo &a);
    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const matrix_dcsr_cpu &m, uint32_t size);
    matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m);
    matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m);
}