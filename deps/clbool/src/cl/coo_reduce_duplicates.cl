#ifndef RUN

#include "clion_defines.cl"

#define GROUP_SIZE 256

#endif


__kernel void scan_blelloch(__local uint *positions) {

    uint local_id = get_local_id(0);
    uint block_size = GROUP_SIZE;
    uint dp = 1;

    for (uint s = block_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        positions[block_size] = positions[block_size - 1];
        positions[block_size - 1] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint s = 1; s < block_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;

            uint t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

}

/* check last element in each group */

__kernel void init_duplicates(
            __global const uint *rows,
            __global const uint *cols,
            __global uint *duplicates_per_tb,
            uint size,
            uint duplicates_size
)  {
    uint global_id = get_global_id(0);
    uint group_length = global_id == duplicates_size - 1 ? (size & (GROUP_SIZE - 1)) : GROUP_SIZE;
    uint idx = global_id * GROUP_SIZE + group_length - 1;

    if (global_id >= duplicates_size) return;

    if (global_id == duplicates_size - 1) {
        duplicates_per_tb[global_id] = 0;
        duplicates_per_tb[duplicates_size] = 0;
    } else {
        duplicates_per_tb[global_id] = rows[idx] == rows[idx + 1] && cols[idx] == cols[idx + 1] ? 1 : 0;
    }
}



/*
 * Having rows and cola arrays after merging, here we reduce duplicates locally in tb.
 *
 */
__kernel void reduce_duplicates_tb(
            __global uint *rows,
            __global uint *cols,
            __global uint *duplicates_per_tb,
            uint size
) {

    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);
    uint group_id = get_group_id(0);
    uint group_shift = group_id * GROUP_SIZE;
    uint last_group_size = size & (GROUP_SIZE - 1);
    uint last_group_id = get_num_groups(0) - 1;

    __local uint rows_local[GROUP_SIZE];
    __local uint cols_local[GROUP_SIZE];
    __local uint positions[GROUP_SIZE + 1];

    if (local_id == 0) positions[GROUP_SIZE] = 0;

    if (global_id < size) {
        rows_local[local_id] = rows[global_id];
        cols_local[local_id] = cols[global_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (global_id >= size) {
        positions[local_id] = 0;
    } else if (local_id == GROUP_SIZE - 1 || global_id == size - 1) {
        positions[local_id] = duplicates_per_tb[group_id] ? 0 : 1;
    } else {
        positions[local_id] = (rows_local[local_id] == rows_local[local_id + 1]
                               && cols_local[local_id] == cols_local[local_id + 1]) ? 0 : 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    scan_blelloch(positions);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (local_id == 0) {
        duplicates_per_tb[group_id] = group_id == last_group_id ?
                last_group_size - positions[GROUP_SIZE] : GROUP_SIZE - positions[GROUP_SIZE];
//        printf("duplicates_per_tb: %d\n"
//               "group_id: %d\n"
//               "last_group_id: %d\n",
//               duplicates_per_tb[group_id],
//               group_id,
//               last_group_id);
    }

    if (global_id < size && (positions[local_id] != positions[local_id + 1])) {
        rows[group_shift + positions[local_id]] = rows_local[local_id];
        cols[group_shift + positions[local_id]] = cols_local[local_id];
    }
}


__kernel void shift_tb(
            __global const uint *rows_old,
            __global const uint *cols_old,

            __global uint *rows_new,
            __global uint *cols_new,

            __global const uint *group_shifts,
            uint size
) {

    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);
    uint group_id = get_group_id(0);
    uint groups_num = get_num_groups(0);
    uint old_group_length = group_id == groups_num - 1 ? (size & (GROUP_SIZE - 1)) : GROUP_SIZE;
    uint group_start = GROUP_SIZE * group_id - group_shifts[group_id];

    uint group_length = old_group_length - (group_shifts[group_id + 1] - group_shifts[group_id]);

    if (local_id < group_length) {
        rows_new[group_start + local_id] = rows_old[global_id];
        cols_new[group_start + local_id] = cols_old[global_id];
    }
}
