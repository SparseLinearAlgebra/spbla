#ifndef RUN

#include "../clion_defines.cl"

#define GROUP_SIZE 32
#define NNZ_ESTIMATION 32
#define TABLE_SIZE 32
#endif

// 4 threads for 4 roes
#define WARP 32 // TODO add define for amd to 64
// how many rows_gpu (tables) can wo process by one threadblock
#define ROWS_PER_TB (GROUP_SIZE / PWARP)
#define HASH_SCAL 107


uint ceil_to_power2(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__kernel void bitonic_sort(__local uint *data,
                           uint size) {

    uint half_segment_length, local_line_id, local_twin_id, group_line_id, line_id, twin_id;
    uint local_id_real = get_local_id(0);

    uint outer = ceil_to_power2(size);
    uint threads_needed = outer / 2;
    // local_id < outer / 2

    uint segment_length = 2;
    while (outer != 1) {
        for (uint local_id = local_id_real; local_id < threads_needed; local_id += GROUP_SIZE) {
            half_segment_length = segment_length / 2;
            // id inside a segment
            local_line_id = local_id & (half_segment_length - 1);
            // index to compare and swap
            local_twin_id = segment_length - local_line_id - 1;
            // segment id
            group_line_id = local_id / half_segment_length;
            // индексы элементов в массиве
            line_id = segment_length * group_line_id + local_line_id;
            twin_id = segment_length * group_line_id + local_twin_id;

            if (line_id < size && twin_id < size && data[line_id] > data[twin_id]) {
                uint tmp = data[line_id];
                data[line_id] = data[twin_id];
                data[twin_id] = tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = half_segment_length; j > 1; j >>= 1) {
            for (uint local_id = local_id_real; local_id < threads_needed; local_id += GROUP_SIZE) {
                uint half_j = j / 2;
                local_line_id = local_id & (half_j - 1);
                local_twin_id = local_line_id + half_j;
                group_line_id = local_id / half_j;
                line_id = j * group_line_id + local_line_id;
                twin_id = j * group_line_id + local_twin_id;

                if (line_id < size && twin_id < size && data[line_id] > data[twin_id]) {
                    uint tmp = data[line_id];
                    data[line_id] = data[twin_id];
                    data[twin_id] = tmp;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        outer >>= 1;
        segment_length <<= 1;
    }
}


uint search_global(__global const uint *array, uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}

void print_array(__local uint *data, uint size) {
    printf("\n");
    for (uint i = 0; i < size; ++i) {
        printf("%d: %d, ", i, data[i]);
    }
    printf("\n");
}

__kernel void hash_symbolic_tb(__global const uint *indices, // indices -- aka premutation
                               uint group_start,
                               uint group_length,

                               __global uint *nnz_estimation, // это нужно обновлять

                               __global const uint *a_rows_pointers,
                               __global const uint *a_cols,

                               __global const uint *b_rows_pointers,
                               __global const uint *b_rows_compressed,
                               __global const uint *b_cols,
                               const uint b_nzr
) {

    uint hash, old, row_index, a_start, a_end, col_index, b_col, b_rpt;
    uint local_id = get_local_id(0); // 0 - 255
    uint row_id_bin = get_group_id(0); // 0, 1, ...
    uint warps_per_group = GROUP_SIZE / WARP; // 256 / 32 -- how many vals of A row you can process in ones
    // от 0 до 31
    uint id_in_warp = get_local_id(0) & (WARP - 1); // 0 - 31 , get_global_id(0) & (WARP - 1) == get_global_id(0) % WARP
    uint warp_id = local_id / WARP; // (0 - 255) / 32 -- warp id of thread
    uint row_pos = group_start + row_id_bin; //

    __local uint hash_table[TABLE_SIZE];
    __local uint nz_count[GROUP_SIZE];
    __local uint *thread_nz = nz_count + local_id;
    thread_nz[0] = 0;

    for (uint i = local_id; i < TABLE_SIZE; i += GROUP_SIZE) {
        hash_table[i] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    row_index = indices[row_pos];
    a_start = a_rows_pointers[row_index];
    a_end = a_rows_pointers[row_index + 1];

    for (uint a_prt = a_start + warp_id; a_prt < a_end; a_prt += warps_per_group) {
        col_index = a_cols[a_prt]; // позицию этого будем искать в матрице B
        b_rpt = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_rpt == b_nzr) {
            continue;
        }

        for (uint k = b_rows_pointers[b_rpt] + id_in_warp; k < b_rows_pointers[b_rpt + 1]; k += WARP) {
            b_col = b_cols[k];
            // Now go to hashtable and search for b_col
            hash = (b_col * HASH_SCAL) & (TABLE_SIZE - 1);
            while (true) {
                if (hash_table[hash] == b_col) {
                    break;
                } else if (hash_table[hash] == -1) {
                    old = atom_cmpxchg(hash_table + hash, -1, b_col);
                    if (old == -1) {
                        thread_nz[0]++;
                        break;
                    }
                } else {
                    hash = (hash + 1) & (TABLE_SIZE - 1);
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
//    if (get_group_id(0) == 0 && local_id == 0) {
//        print_array(hash_table, TABLE_SIZE);
//    }
    // reduce nz values
    int step = GROUP_SIZE / 2;
    while (step > 0) {
        if (local_id < step) {
            nz_count[local_id] = nz_count[local_id] + nz_count[local_id + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        step /= 2;
    }

    if (local_id == 0) {
//        if (get_group_id(0) == 0) {
//            printf("\nNZ: %d\n", nz_count[0]);
//        }
        nnz_estimation[row_index] = nz_count[0];
    }
}

__kernel void hash_numeric_tb(__global const uint *indices, // indices -- aka premutation
                              uint group_start,

                              __global
                              const uint *pre_matrix_rows_pointers,
                              __global uint *c_cols,

                              __global const uint *a_rows_pointers,
                              __global const uint *a_cols,

                              __global const uint *b_rows_pointers,
                              __global const uint *b_rows_compressed,
                              __global const uint *b_cols,
                              const uint b_nzr
) {

    uint hash, old, row_index, a_start, a_end, col_index, b_col, b_rpt;
    uint local_id = get_local_id(0); // 0 - 255
    uint row_id_bin = get_group_id(0); // 0, 1, ...
    uint warps_per_group = GROUP_SIZE / WARP; // 256 / 32 -- how many vals of A row you can process in ones
    // от 0 до 31
    uint id_in_warp = get_local_id(0) & (WARP - 1); // 0 - 31 , get_global_id(0) & (WARP - 1) == get_global_id(0) % WARP
    uint warp_id = local_id / WARP; // (0 - 255) / 32 -- warp id of thread
    uint row_pos = group_start + row_id_bin; //


    row_index = indices[row_pos];
    a_start = a_rows_pointers[row_index];
    a_end = a_rows_pointers[row_index + 1];

    uint c_row_start = pre_matrix_rows_pointers[row_index];
    uint c_row_end = pre_matrix_rows_pointers[row_index + 1];
    uint c_row_length = c_row_end - c_row_start;

    if(c_row_length == 0) return;

    __local uint hash_table[TABLE_SIZE];

    for (uint i = local_id; i < TABLE_SIZE; i += GROUP_SIZE) {
        hash_table[i] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint a_prt = a_start + warp_id; a_prt < a_end; a_prt += warps_per_group) {
        col_index = a_cols[a_prt]; // позицию этого будем искать в матрице B
        b_rpt = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_rpt == b_nzr) {
            continue;
        }

        for (uint k = b_rows_pointers[b_rpt] + id_in_warp; k < b_rows_pointers[b_rpt + 1]; k += WARP) {
            b_col = b_cols[k];
            // Now go to hashtable and search for b_col
            hash = (b_col * HASH_SCAL) & (TABLE_SIZE - 1);
            while (true) {
                if (hash_table[hash] == b_col) {
                    break;
                } else if (hash_table[hash] == -1) {
                    old = atom_cmpxchg(hash_table + hash, -1, b_col);
                    if (old == -1) {
                        hash_table[hash] = b_col;
                        break;
                    }
                } else {
                    hash = (hash + 1) & (TABLE_SIZE - 1);
                }
            }
        }
    }


    barrier(CLK_LOCAL_MEM_FENCE);
//    if (get_group_id(0) == 0 && local_id == 0) {
//        printf("\nBEFORE SORT\n");
//        printf("\nc_row_length: %d\n", c_row_length);
//        printf("\nTABLE_SIZE: %d\n", TABLE_SIZE);
//
//        print_array(hash_table, TABLE_SIZE);
//    }
    __global uint *c_cols_cur_global = c_cols + pre_matrix_rows_pointers[row_index];
    // sort values and save
    barrier(CLK_LOCAL_MEM_FENCE);
    bitonic_sort(hash_table, TABLE_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
//    if (get_group_id(0) == 0 && local_id == 0) {
//        printf("AFTER SORT");
//        print_array(hash_table, TABLE_SIZE);
//    }
    for (uint i = local_id; i < c_row_length; i += GROUP_SIZE) {
        c_cols_cur_global[i] = hash_table[i];
    }

}