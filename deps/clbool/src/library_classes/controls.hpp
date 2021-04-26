#pragma once

#include "../common/cl_includes.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

// TODO: define warp size based on GPU vendor
#define WARP_SIZE 32

namespace clbool {
    struct Controls {
        const cl::Device device;
        const cl::Context context;
        cl::CommandQueue queue;
        cl::CommandQueue async_queue;
        const uint32_t block_size = uint32_t(256);

        Controls(cl::Device device) :
                device(device)
        , context(cl::Context(device))
        , queue(cl::CommandQueue(context))
        , async_queue(cl::CommandQueue(context, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
        {}

        cl::Program create_program_from_source(const char * kernel, uint32_t length) const {
            return cl::Program(context, {{kernel, length}});
        }
    };
}

