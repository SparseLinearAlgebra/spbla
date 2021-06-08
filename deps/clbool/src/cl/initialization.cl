#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif


__kernel void fill_with(
        __global uint* array,
        uint size,
        uint val
        ) {
    uint global_id = get_global_id(0);
    if (global_id >= size) return;
    array[global_id] = val;
}
