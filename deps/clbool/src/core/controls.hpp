#pragma once

#include "../common/cl_includes.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#define WARP_SIZE 32

namespace clbool {

    struct Controls {
        const cl::Device device;
        const cl::Context context;
        cl::CommandQueue queue;
        cl::CommandQueue async_queue;
        uint32_t max_wg_size;
        const uint32_t block_size = uint32_t(256);

        Controls(cl::Device device, uint32_t max_wg_size = 256) :
                device(device)
        , context(cl::Context(device))
        , queue(cl::CommandQueue(context))
        , async_queue(cl::CommandQueue(context, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
        , max_wg_size(max_wg_size)
        {}

    };
}

