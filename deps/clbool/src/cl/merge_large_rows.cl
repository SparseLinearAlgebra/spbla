#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

#define BUFFER_SIZE 256

#define SWAP_LOCAL(a, b) {__local uint * tmp=a; a=b; b=tmp;}
#define SWAP_GLOBAL(a, b) {__global uint * tmp=a; a=b; b=tmp;}

inline uint
search_global(__global const unsigned int *array, uint value, uint size) {
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


inline uint
search_local(__local const unsigned int *array, uint value, uint size) {
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


inline void
scan(__local uint *positions, uint scan_size) {
    uint local_id = get_local_id(0);
    uint dp = 1;

    for (uint s = scan_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }
        dp <<= 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == scan_size - 1) {
        positions[scan_size] = positions[local_id];
        positions[local_id] = 0;
    }

    for (uint s = 1; s < scan_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;

            unsigned int t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void
set_positions(__local const uint *positions, __local const uint *vals, __local uint *result, uint size) {
    uint local_id = get_local_id(0);

    if (local_id >= size) return;
    if (positions[local_id] != positions[local_id + 1]) {
        result[positions[local_id]] = vals[local_id];
    }
}


inline void
merge_local(__local const uint *a, __local const uint *b, __local uint *c, uint sizeA, uint sizeB) {
    uint diag_index = get_local_id(0);
    uint res_size = sizeA + sizeB;
    if (diag_index >= res_size) return;
    uint min_side = sizeA < sizeB ? sizeA : sizeB;
    uint max_side = res_size - min_side;

    uint diag_length = diag_index < min_side ? diag_index + 2 :
                               diag_index < max_side ? min_side + 1 :
                               res_size - diag_index;

    uint r = diag_length, l = 0, m = 0;
    uint below_idx_a = 0, below_idx_b = 0, above_idx_a = 0, above_idx_b = 0;
    uint above = 0, below = 0;

    while (l < r) {
        m = (l + r) / 2;
        below_idx_a = diag_index < sizeA ? diag_index - m + 1 : sizeA - m;
        below_idx_b = diag_index < sizeA ? m - 1 : (diag_index - sizeA) + m;

        above_idx_a = below_idx_a - 1;
        above_idx_b = below_idx_b + 1;

        below = m == 0 ? 1 : a[below_idx_a] > b[below_idx_b];
        above = m == diag_length - 1 ? 0 : a[above_idx_a] > b[above_idx_b];

        // success
        if (below != above) {
            if (((diag_index < sizeA) && m == 0) || below_idx_b >= sizeB) {
                c[diag_index] = a[above_idx_a];
                return;
            }
            if (((diag_index < sizeB) && m == diag_length - 1) || above_idx_a >= sizeA)  {
                c[diag_index] = b[below_idx_b];
                return;
            }

            c[diag_index] =  max(a[above_idx_a], b[below_idx_b]);
            return;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}


// тут нам нужны оба указателия, поэтому вернем примерно sizeB * above_idx_a + above_idx_b
inline uint
merge_pointer_global(__global const uint *a, __local const uint *b, __global uint *c, uint sizeA, uint sizeB, uint diag_index) {
    unsigned int sizeB_inc = sizeB + 1;
    unsigned int res_size = sizeA + sizeB;
    unsigned int min_side = sizeA < sizeB ? sizeA : sizeB;
    unsigned int max_side = res_size - min_side;

    unsigned int diag_length = diag_index < min_side ? diag_index + 2 :
                               diag_index < max_side ? min_side + 1:
                               res_size - diag_index;

    unsigned r = diag_length;
    unsigned l = 0;
    unsigned int m = 0;

    unsigned int below_idx_a = 0;
    unsigned int below_idx_b = 0;
    unsigned int above_idx_a = 0;
    unsigned int above_idx_b = 0;

    unsigned int above = 0;
    unsigned int below = 0;

    while (true) {
        m = (l + r) / 2;
        below_idx_a = diag_index < sizeA ? diag_index - m + 1 : sizeA - m;
        below_idx_b = diag_index < sizeA ? m - 1 : (diag_index - sizeA) + m;

        above_idx_a = below_idx_a - 1;
        above_idx_b = below_idx_b + 1;

        below = m == 0 ? 1 : a[below_idx_a] > b[below_idx_b];
        above = m == diag_length - 1 ? 0 : a[above_idx_a] > b[above_idx_b];

        // success
        if (below != above) {
            if (((diag_index < sizeA) && m == 0) || below_idx_b >= sizeB) {
                return sizeB_inc * above_idx_a + above_idx_b;
            }
            if (((diag_index < sizeB) && m == diag_length - 1) || above_idx_a >= sizeA) {
                return sizeB_inc * below_idx_a + below_idx_b;
            }

            return a[above_idx_a] > b[below_idx_b] ? above_idx_a * sizeB_inc + above_idx_b :
                                                        below_idx_a * sizeB_inc + below_idx_b;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}


inline void
merge_global(__global const uint *a, __local const uint *b, __global uint *c, uint sizeA, uint sizeB) {
    uint sizeB_inc = sizeB + 1;
    uint step_length = ((sizeA + sizeB) + GROUP_SIZE - 1) / GROUP_SIZE;
    uint diag_index = get_local_id(0) * step_length;
    if (diag_index >= sizeA + sizeB) return;
    uint m_ptr = merge_pointer_global(a, b, c, sizeA, sizeB, diag_index);
    uint a_ptr = m_ptr / sizeB_inc;
    uint b_ptr = m_ptr % sizeB_inc;

    for (uint i = 0; i < step_length; ++i) {
        if ((a_ptr < sizeA && a[a_ptr] < b[b_ptr]) || b_ptr >= sizeB) {
            c[diag_index + i] = a[a_ptr++];
            continue;
        }

        if (b_ptr < sizeB) {
            c[diag_index + i] = b[b_ptr++];
            continue;
        }
        return;
    }
}


inline uint
ceil_to_power2(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


__kernel void merge_large_rows(__global const uint *indices,
                               uint group_start, // indices_pointers[workload_group_id]

                               __global uint *aux_mem_pointers,
                               __global uint *aux_mem,

                               __global const uint *pre_matrix_rows_pointers,
                               __global uint *pre_matrix_cols_indices,
                               __global uint *nnz_estimation,

                               __global const uint *a_rows_pointers,
                               __global const uint *a_cols,

                               __global const uint *b_rows_pointers,
                               __global const uint *b_rows_compressed,
                               __global const uint *b_cols,
                               const uint b_nzr
) {
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);

    uint col_index, b_row_pointer, b_start, b_end, b_left_to_read, scan_size, new_length;

    __local uint merge_buffer1[BUFFER_SIZE];
    __local uint merge_buffer2[BUFFER_SIZE];

    __local uint *incoming_row = merge_buffer1;
    __local uint *local_res = merge_buffer2;
    __local uint *local_tmp;

    __local uint positions[BUFFER_SIZE + 1];

    uint fill_pointer = 0;

    uint row_index = indices[group_start + group_id];
    uint a_start = a_rows_pointers[row_index];
    uint a_end = a_rows_pointers[row_index + 1];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[row_index];
    __global uint *current_row_aux_memory = aux_mem + aux_mem_pointers[group_id];

    __global uint *buff_1_global = result;
    __global uint *buff_2_global = current_row_aux_memory;
    __global uint *global_tmp;

    __local bool global_flag;
    __local uint new_b_row_start;
    __local uint old_b_row_end;
    uint global_fill_pointer = 0;

    if (local_id == 0) {
        global_flag = false;
        new_b_row_start = 0;
        old_b_row_end = 0;
    }

    for (uint a_row_pointer = a_start; a_row_pointer < a_end; ++a_row_pointer) {
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (!global_flag) {
            col_index = a_cols[a_row_pointer];
            b_row_pointer = search_global(b_rows_compressed, col_index, b_nzr);
            if (b_row_pointer == b_nzr) continue;
        }

        b_start = global_flag ? new_b_row_start : b_rows_pointers[b_row_pointer];
        b_end =  global_flag ? old_b_row_end : b_rows_pointers[b_row_pointer + 1];
        b_left_to_read = b_end - b_start;

        barrier(CLK_LOCAL_MEM_FENCE); // this barrier is very important, because we use global_flag in the upper lines
        if (global_flag && local_id == 0) {
            global_flag = false;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ------------------ COPY  ---------------------------
        uint elem_id = b_start + local_id;
        if (elem_id < b_end) {
            uint fill_position = local_id + fill_pointer;
            // fill_position нам тут нужен для проверки, что ряд поместится в local_res при потенциальном копировании.
            if (fill_position < BUFFER_SIZE) {
                if (fill_pointer == 0) {
                    local_res[local_id] = b_cols[elem_id];
                } else {
                    incoming_row[local_id] = b_cols[elem_id];
                    positions[local_id] =
                            search_local(local_res, incoming_row[local_id], fill_pointer) == fill_pointer ? 1 : 0;
                }
            } else {
                if (fill_position == BUFFER_SIZE) {
                    global_flag = true;
                    new_b_row_start = elem_id;
                    old_b_row_end = b_end;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (b_start + BUFFER_SIZE < b_end && !global_flag && local_id == 0) {
            global_flag = true;
            new_b_row_start = b_start + BUFFER_SIZE;
            old_b_row_end = b_end;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // если кто-то из потоков не смог записать своё значение из второй матрицы, нужно будет вернуться к этому ряду
        if (global_flag) {a_row_pointer --;}

        // new_b_row_start -  первая позиция, которую не успели записать
        uint filled_b_length = global_flag ? new_b_row_start - b_start : b_left_to_read;

        // ------------------ LOCAL MERGE ---------------------------
        if (fill_pointer != 0) {
            // для всех кто не успел себя записать, позиции обулим
            if (local_id >= filled_b_length) positions[local_id] = 0;
            scan_size = ceil_to_power2(filled_b_length);

            scan(positions, scan_size);
            barrier(CLK_LOCAL_MEM_FENCE);

            new_length = positions[filled_b_length];

            set_positions(positions, incoming_row, local_res + fill_pointer, filled_b_length);
            barrier(CLK_LOCAL_MEM_FENCE);

            merge_local(local_res, local_res + fill_pointer, incoming_row, fill_pointer, new_length);
            barrier(CLK_LOCAL_MEM_FENCE);

            SWAP_LOCAL(incoming_row, local_res);
        } else {
            new_length = filled_b_length;
        }

        fill_pointer += new_length;
        bool last_step = (a_row_pointer == a_end - 1) && (filled_b_length == b_left_to_read);

        // ------------------ GLOBAL MERGE ---------------------------

        if (global_flag || last_step) {
            if (global_fill_pointer == 0) {
                if (local_id < fill_pointer) {
                    result[local_id] = local_res[local_id];
                }
                new_length = fill_pointer;
            }
            else {
                if (local_id < fill_pointer) {
                    positions[local_id] =
                            search_global(buff_1_global, local_res[local_id], global_fill_pointer) == global_fill_pointer
                            ? 1 : 0;
                } else {
                    positions[local_id] = 0;
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);

                scan_size = ceil_to_power2(fill_pointer);
                scan(positions, scan_size);
                barrier(CLK_LOCAL_MEM_FENCE);

                new_length = positions[fill_pointer];
                // переместим их из buff2 в buff1
                set_positions(positions, local_res, incoming_row, fill_pointer);
                barrier(CLK_LOCAL_MEM_FENCE);

                merge_global(buff_1_global, incoming_row, buff_2_global, global_fill_pointer, new_length);
                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);
                SWAP_GLOBAL(buff_1_global, buff_2_global);
            }

            global_fill_pointer += new_length;
            fill_pointer = 0;

        }
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }


    if (result != buff_1_global) {
        uint steps = (global_fill_pointer + GROUP_SIZE - 1) / GROUP_SIZE;
        for (uint step = 0; step < steps; ++step) {
            uint elem_id = local_id + GROUP_SIZE * step;
            if (elem_id < global_fill_pointer) {
                result[elem_id] = buff_1_global[elem_id];
            }
        }
    }

    if (local_id == 0) {
        nnz_estimation[row_index] = global_fill_pointer;
    }
}