#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

inline
uint merge_pointer(__global const uint *A,
                   __global const uint *B,
                   uint diag_index,
                   uint sizeA,
                   uint sizeB) {

    uint res_size = sizeA + sizeB;
    uint min_side = sizeA < sizeB ? sizeA : sizeB;
    uint max_side = res_size - min_side;

    uint diag_length = diag_index < min_side ? diag_index + 2 :
                               diag_index < max_side ? min_side + 1 :
                               res_size - diag_index;
    uint r = diag_length;
    uint l = 0;
    uint m = 0;

    uint below_idx_a = 0;
    uint below_idx_b = 0;
    uint above_idx_a = 0;
    uint above_idx_b = 0;

    uint above = 0; // значение сравнения справа сверху
    uint below = 0; // значение сравнения слева снизу


    while (true) {
        m = (l + r) / 2;
        below_idx_a = diag_index < sizeA ? diag_index - m + 1 : sizeA - m;
        below_idx_b = diag_index < sizeA ? m - 1 : (diag_index - sizeA) + m;

        above_idx_a = below_idx_a - 1;
        above_idx_b = below_idx_b + 1;

        below = m == 0 ? 1 : A[below_idx_a] > B[below_idx_b];
        above = m == diag_length - 1 ? 0 : A[above_idx_a] > B[above_idx_b];

        // success
        if (below != above) {
            return m;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}

inline
void merge_local(__global uint *C,
                 __local const uint *A,
                 __local const uint *B,
                 uint diag_index,
                 uint sizeA,
                 uint sizeB) {
    uint real_diag = get_global_id(0);
    uint res_size = sizeA + sizeB;
    uint min_side = sizeA < sizeB ? sizeA : sizeB;
    uint max_side = res_size - min_side;

    uint diag_length = diag_index < min_side ? diag_index + 2 :
                       diag_index < max_side ? min_side + 1 :
                       res_size - diag_index;
    uint r = diag_length;
    uint l = 0;
    uint m = 0;

    uint below_idx_a = 0;
    uint below_idx_b = 0;
    uint above_idx_a = 0;
    uint above_idx_b = 0;

    uint above = 0; // значение сравнения справа сверху
    uint below = 0; // значение сравнения слева снизу


    while (true) {
        m = (l + r) / 2;
        below_idx_a = diag_index < sizeA ? diag_index - m + 1 : sizeA - m;
        below_idx_b = diag_index < sizeA ? m - 1 : (diag_index - sizeA) + m;

        above_idx_a = below_idx_a - 1;
        above_idx_b = below_idx_b + 1;

        below = m == 0 ? 1 : A[below_idx_a] > B[below_idx_b];
        above = m == diag_length - 1 ? 0 : A[above_idx_a] > B[above_idx_b];
        //is_greater_local(rowsA, colsA, rowsB, colsB, above_idx_a, above_idx_b);


        // success
        if (below != above) {

            if (((diag_index < sizeA) && m == 0) || below_idx_b >= sizeB) {
                C[real_diag] = A[above_idx_a];
                return;
            }
            if (((diag_index < sizeB) && m == diag_length - 1) || above_idx_a >= sizeA) {
                C[real_diag] = B[below_idx_b];
                return;
            }
            // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю

            C[real_diag] = A[above_idx_a] > B[below_idx_b] ? A[above_idx_a] : B[below_idx_b];

            return;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}

inline
uint get_a_index(uint diag_index, uint m, uint sizeA) {
    if (diag_index < sizeA) {
        return diag_index - m + 1;
    }
    return sizeA - m;
}

inline
uint get_b_index(uint diag_index, uint m, uint sizeA) {
    if (diag_index < sizeA) {
        return m;
    }
    return (diag_index - sizeA) + m + 1;
}

__kernel void merge(__global uint *C,
                    __global const uint *A,
                    __global const uint *B,
                    uint sizeA,
                    uint sizeB) {

    uint diag_index = get_global_id(0);
    uint num_groups = get_num_groups(0);
    uint group_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);
    uint group_size = get_local_size(0);

    __local uint ab_local[GROUP_SIZE];

    __local uint a_start, a_end, b_start, b_end;
    if (local_id == 0) {
        a_start = 0; a_end = sizeA;  b_start = 0; b_end = sizeB;
    }
    uint m;
    uint res_size =  sizeA + sizeB;
    if (group_id == 0) {
        if (local_id == 0) {
            diag_index = min(group_size - 1, res_size - 1);
            m = merge_pointer(A, B, diag_index, sizeA, sizeB);
            a_end = get_a_index(diag_index, m, sizeA);
            b_end = get_b_index(diag_index, m, sizeA);
        }
    } else if (group_id == num_groups-1) {
        if (local_id == 0) {
            diag_index = global_id - 1;
            m = merge_pointer(A, B, global_id - 1, sizeA, sizeB);
            a_start = get_a_index(diag_index, m, sizeA);
            b_start = get_b_index(diag_index, m, sizeA);
        }
    } else {
        if (local_id < 2) {
            diag_index = global_id + local_id * (GROUP_SIZE - 1) - 1 ;
            m = merge_pointer(A, B, diag_index, sizeA, sizeB);
            if (local_id == 0) {
                a_start = get_a_index(diag_index, m, sizeA);
                b_start = get_b_index(diag_index, m, sizeA);
            } else {
                a_end = get_a_index(diag_index, m, sizeA);
                b_end = get_b_index(diag_index, m, sizeA);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint a_local_size = a_end - a_start;
    uint b_local_size = b_end - b_start;

    __local uint *a_local = ab_local;
    __local uint *b_local = a_local + a_local_size;

    if (local_id < a_local_size) {
        ab_local[local_id] = A[a_start + local_id];
    }
    else if (local_id - a_local_size < b_local_size) {
        ab_local[local_id] = B[b_start + local_id - a_local_size];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    diag_index = get_global_id(0);
    if (diag_index >= sizeA + sizeB) return;

    merge_local(C, a_local,b_local, get_local_id(0), a_local_size, b_local_size);

}



