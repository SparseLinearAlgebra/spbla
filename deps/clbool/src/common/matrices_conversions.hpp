#pragma once

#include "utils.hpp"
#include "../core/kernel.hpp"
#include "cl_operations.hpp"


namespace clbool {
    matrix_coo dcsr_to_coo_shallow(Controls &controls, matrix_dcsr &a);
    matrix_coo dcsr_to_coo_deep(Controls &controls, const matrix_dcsr &a);
    matrix_dcsr coo_to_dcsr_shallow(Controls &controls, const matrix_coo &a);

    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const matrix_dcsr_cpu &m, uint32_t size);

    matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m);
    matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m);
    matrix_csr_cpu matrix_csr_from_gpu(Controls &controls, const matrix_csr &m);

    matrix_csr_cpu csr_cpu_from_pairs(const matrix_coo_cpu_pairs &mat, uint32_t m, uint32_t n);
    matrix_csr csr_from_cpu(Controls &controls, const matrix_csr_cpu &m);
    matrix_csr_cpu csr_cpu_from_coo_cpu(const matrix_coo_cpu &mat, uint32_t m, uint32_t n);

    // shallow for cols!
    matrix_dcsr csr_to_dcsr(Controls &controls, const matrix_csr &m);
    // shallow for cols!
    matrix_csr dcsr_to_csr(Controls &controls, const matrix_dcsr &m);
}