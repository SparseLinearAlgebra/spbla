#pragma once

#include "../common/cl_includes.hpp"
#include "../common/utils.hpp"
#include "../core/controls.hpp"
#include "../core/kernel.hpp"

namespace clbool::coo{
    void sort_arrays(Controls& controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n);
}



