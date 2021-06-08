#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

bool is_greater_local(__local const uint* rows,
                      __local const uint* cols,
                      uint line_id,
                      uint twin_id) {

    return (rows[line_id] > rows[twin_id]) ||
            ((rows[line_id] == rows[twin_id]) && (cols[line_id] > cols[twin_id]));
}

bool is_greater_global(__global const uint* rows,
                       __global const uint* cols,
                        uint line_id,
                        uint twin_id) {

    return (rows[line_id] > rows[twin_id]) ||
           ((rows[line_id] == rows[twin_id]) && (cols[line_id] > cols[twin_id]));
}

//void print_array(__local uint *data, uint size) {
//
//    for (uint i = 0; i < size; ++i) {
//        printf("(%d:  %d),  ", i, data[i]);
//    }
//    printf("\n");
//}


__kernel void local_bitonic_begin(__global uint* rows,
                                  __global uint* cols,
                                  uint n) {

    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint global_id = get_global_id(0);
    uint work_size = GROUP_SIZE * 2;
//    if (global_id == 0) {
//        printf("GROUP_SIZE: %d\n", GROUP_SIZE);
//        printf("get_global_size: %d\n", get_global_size(0));
//        printf("get_num_groups: %d\n", get_num_groups(0));
//    }
    __local uint local_rows[GROUP_SIZE * 2];
    __local uint local_cols[GROUP_SIZE * 2];

    uint tmp_row = 0;
    uint tmp_col = 0;

    uint read_idx = work_size * group_id + local_id;

    local_cols[local_id] = read_idx < n ? cols[read_idx] : -1;
    local_rows[local_id] = read_idx < n ? rows[read_idx] : -1;

    read_idx += GROUP_SIZE;

    local_cols[local_id + GROUP_SIZE] = read_idx < n ? cols[read_idx] : -1;
    local_rows[local_id + GROUP_SIZE] = read_idx < n ? rows[read_idx] : -1;

    barrier(CLK_LOCAL_MEM_FENCE);
    uint segment_length = 1;
    while (segment_length < work_size) {
        segment_length <<= 1;
        uint local_line_id = local_id % (segment_length / 2);
        uint local_twin_id = segment_length - local_line_id - 1;
        uint group_line_id = local_id / (segment_length / 2);
        uint line_id = segment_length * group_line_id + local_line_id;
        uint twin_id = segment_length * group_line_id + local_twin_id;

        if (is_greater_local(local_rows, local_cols, line_id, twin_id)) {
            tmp_row = local_rows[line_id];
            tmp_col = local_cols[line_id];

            local_rows[line_id] = local_rows[twin_id];
            local_cols[line_id] = local_cols[twin_id];

            local_rows[twin_id] = tmp_row;
            local_cols[twin_id] = tmp_col;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = segment_length / 2; j > 1; j >>= 1) {
            local_line_id = local_id % (j / 2);
            local_twin_id = local_line_id + (j / 2);
            group_line_id = local_id / (j / 2);
            line_id = j * group_line_id + local_line_id;
            twin_id = j * group_line_id + local_twin_id;
            if (is_greater_local(local_rows, local_cols, line_id, twin_id)) {
                tmp_row = local_rows[line_id];
                tmp_col = local_cols[line_id];

                local_rows[line_id] = local_rows[twin_id];
                local_cols[line_id] = local_cols[twin_id];

                local_rows[twin_id] = tmp_row;
                local_cols[twin_id] = tmp_col;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

//    uint glob_id = get_global_id(0);
//
//    if (glob_id == 0) {
//        print_array(local_rows, work_size);
//    }

    uint write_idx = work_size * group_id + local_id;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id];
        rows[write_idx] = local_rows[local_id];
    }

    write_idx += GROUP_SIZE;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id + GROUP_SIZE];
        rows[write_idx] = local_rows[local_id + GROUP_SIZE];
    }
}


__kernel void bitonic_global_step(__global uint* rows,
                                  __global uint* cols,
                                  uint segment_length,
                                  uint mirror,
                                  uint n)
{
    uint global_id = get_global_id(0);
    uint group_id = get_group_id(0);
    uint local_line_id = global_id % (segment_length / 2);
    uint local_twin_id = mirror ? segment_length - local_line_id - 1 : local_line_id + (segment_length / 2);
    uint group_line_id = global_id / (segment_length / 2);
    uint line_id = segment_length * group_line_id + local_line_id;
    uint twin_id = segment_length * group_line_id + local_twin_id;
//    uint max_id = (get_local_size(0) - 1);
//    if (twin_id == (1024 / 2)) {
//        printf("get global id: %d\n", get_global_id(0));
//    }
    uint tmp_row = 0;
    uint tmp_col = 0;
//    if (group_id  == 359) {
//        printf("fine, twin_id: %d\n", twin_id, n);
//    }
    if ((twin_id < n) && is_greater_global(rows, cols, line_id, twin_id)) {

        tmp_row = rows[line_id];
        tmp_col = cols[line_id];

        rows[line_id] = rows[twin_id];
        cols[line_id] = cols[twin_id];

        rows[twin_id] = tmp_row;
        cols[twin_id] = tmp_col;
    }
}

__kernel void bitonic_local_endings(__global uint* rows,
                                    __global uint* cols,
                                    uint n)
{
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint work_size = GROUP_SIZE * 2;

    __local uint local_rows[GROUP_SIZE * 2];
    __local uint local_cols[GROUP_SIZE * 2];

    uint tmp_row = 0;
    uint tmp_col = 0;

    uint read_idx = work_size * group_id + local_id;

    local_cols[local_id] = read_idx < n ? cols[read_idx] : -1;
    local_rows[local_id] = read_idx < n ? rows[read_idx] : -1;

    read_idx += GROUP_SIZE;

    local_cols[local_id + GROUP_SIZE] = read_idx < n ? cols[read_idx] : -1;
    local_rows[local_id + GROUP_SIZE] = read_idx < n ? rows[read_idx] : -1;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint segment_length = work_size;

    for (uint j = segment_length; j > 1; j >>= 1) {
        uint local_line_id = local_id % (j / 2);
        uint local_twin_id = local_line_id + (j / 2);
        uint group_line_id = local_id / (j / 2);
        uint line_id = j * group_line_id + local_line_id;
        uint twin_id = j * group_line_id + local_twin_id;

        if (is_greater_local(local_rows, local_cols, line_id, twin_id)) {
            tmp_row = local_rows[line_id];
            tmp_col = local_cols[line_id];

            local_rows[line_id] = local_rows[twin_id];
            local_cols[line_id] = local_cols[twin_id];

            local_rows[twin_id] = tmp_row;
            local_cols[twin_id] = tmp_col;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    uint write_idx = work_size * group_id + local_id;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id];
        rows[write_idx] = local_rows[local_id];
    }

    write_idx += GROUP_SIZE;
    if (write_idx < n) {
        cols[write_idx] = local_cols[local_id + GROUP_SIZE];
        rows[write_idx] = local_rows[local_id + GROUP_SIZE];
    }
}



