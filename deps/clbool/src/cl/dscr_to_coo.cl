#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

__kernel void dscr_to_coo(__global const uint* a_rpt_dcsr,
                          __global const uint* a_rows_dcsr,
                          __global uint* c_rows_coo
                          ) {
    uint group_size = get_local_size(0);
    uint group_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint row_start = a_rpt_dcsr[group_id];
    uint row_end = a_rpt_dcsr[group_id + 1];
    uint row = a_rows_dcsr[group_id];
    uint row_length = row_end - row_start;
    for (uint i = local_id; i < row_length; i += group_size) {
        uint elem_id = row_start + i;
        c_rows_coo[elem_id] = row;
    }
}
