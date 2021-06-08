#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

__kernel void dcsr_to_csr_set_size(
        __global const uint *dcsr_rpt,
        __global const uint *dcsr_rows,
        uint nzr, // number of nonzero rows

        __global uint* csr_rpt
        ) {
    uint global_id = get_global_id(0);
    if (global_id >= nzr) return;
    csr_rpt[dcsr_rows[global_id]] = dcsr_rpt[global_id + 1] - dcsr_rpt[global_id];
}

