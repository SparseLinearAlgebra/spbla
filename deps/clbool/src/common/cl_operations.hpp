#pragma once

#include "controls.hpp"
#include "matrix_coo.hpp"
#include "matrix_dcsr.hpp"
#include "kernel.hpp"

namespace clbool {
    void prefix_sum(Controls &controls,
                    cl::Buffer &array,
                    uint32_t &total_sum,
                    uint32_t array_size);
}