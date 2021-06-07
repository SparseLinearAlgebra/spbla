#pragma once
#include "utils.hpp"

namespace clbool {
    Controls create_controls(uint32_t platform_id = 0, uint32_t device_id = 0);
    void show_devices();
}