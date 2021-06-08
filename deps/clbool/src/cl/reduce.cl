#ifndef RUN

#include "clion_defines.cl"

#define GROUP_SIZE 256

#endif

// N of threads = size
__kernel void set_rpt_and_cols(__global uint *rpt,
                               __global uint *cols,
                               uint size) {
    uint global_id = get_global_id(0);
    if (global_id < size) {
        cols[global_id] = 0;
        rpt[global_id] = global_id;
    }
    if (global_id == 0) {
        rpt[size] = size;
    }
}