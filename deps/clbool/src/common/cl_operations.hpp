#pragma once

#include "../core/controls.hpp"
#include "../core/matrix_coo.hpp"
#include "../core/matrix_dcsr.hpp"
#include "../core/kernel.hpp"

namespace clbool {
    void prefix_sum(Controls &controls,
                    cl::Buffer &array,
                    uint32_t &total_sum,
                    uint32_t array_size);

    void fill_with(Controls &controls, cl::Buffer &array, uint32_t size, uint32_t value);
    void fill_with_zeroes(Controls &controls, cl::Buffer &array, uint32_t size);
}