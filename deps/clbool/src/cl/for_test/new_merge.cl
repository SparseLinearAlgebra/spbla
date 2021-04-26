#ifndef RUN
#include "../clion_defines.cl"
#endif

#define SWAP(a, b) {__local uint * tmp=a; a=b; b=tmp;}

#define GROUP_SIZE 256



#define BUFFER_SIZE 256


// тут нам нужны оба указателия, поэтому вернем примерно sizeB * above_idx_a + above_idx_b
inline uint
merge_pointer_global(__global const uint *a, __local const uint *b, __global uint *c, uint sizeA, uint sizeB, uint diag_index) {
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

    unsigned int above = 0; // значение сравнения справа сверху
    unsigned int below = 0; // значение сравнения слева снизу

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
            if ((diag_index < sizeA) && m == 0) {
                return (sizeB + 1) * above_idx_a + above_idx_b;
            }
            if ((diag_index < sizeB) && m == diag_length - 1) {
                return (sizeB + 1) * below_idx_a + below_idx_b;
            }
            // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
            if (a[above_idx_a] > b[below_idx_b]) {
                return above_idx_a * (sizeB + 1) + above_idx_b;
            }

            return below_idx_a * (sizeB + 1) + below_idx_b;
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
    uint step_length = ((sizeA + sizeB) + GROUP_SIZE - 1) / GROUP_SIZE;
    uint diag_index = get_local_id(0) * step_length;
    if (diag_index >= sizeA + sizeB) return;
    uint mptr = merge_pointer_global(a, b, c, sizeA, sizeB, diag_index);
    uint a_ptr = mptr / (sizeB + 1);
    uint b_ptr = mptr % (sizeB + 1);

    for (uint i = 0; i < step_length; ++i) {

        if (a_ptr < sizeA && a[a_ptr] < b[b_ptr] || b_ptr >= sizeB) {
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

__kernel void new_merge(__global const uint *a,
                        __global const uint *b,
                        __global uint *c,
                        uint sizeA,
                        uint sizeB) {

    uint local_id = get_local_id(0);

    __local uint local_buff[GROUP_SIZE];
    local_buff[local_id] = b[local_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    merge_global(a, local_buff, c, sizeA, GROUP_SIZE);
}

inline uint
merge_pointer_full(__global const uint *a, __global const uint *b, __global uint *c, uint sizeA, uint sizeB,
                   uint diag_index) {

    unsigned int res_size = sizeA + sizeB;
    unsigned int min_side = sizeA < sizeB ? sizeA : sizeB;
    unsigned int max_side = res_size - min_side;



    unsigned int diag_length = diag_index < min_side ? diag_index + 2 :
                               diag_index < max_side ? min_side + 2 :
                               res_size - diag_index;

    unsigned r = diag_length;
    unsigned l = 0;
    unsigned int m = 0;

    unsigned int below_idx_a = 0;
    unsigned int below_idx_b = 0;
    unsigned int above_idx_a = 0;
    unsigned int above_idx_b = 0;

    unsigned int above = 0; // значение сравнения справа сверху
    unsigned int below = 0; // значение сравнения слева снизу


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
            if ((diag_index < sizeA) && m == 0) {
                return above_idx_a;
            }
            if ((diag_index < sizeB) && m == diag_length - 1) {
                return below_idx_b + sizeA;
            }
            return a[above_idx_a] > b[below_idx_b] ? above_idx_a : sizeA + below_idx_b;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}


inline void
merge_global_full(__global const uint *a, __global const uint *b, __global uint *c, uint sizeA, uint sizeB) {
    unsigned int diag_index = get_global_id(0);
    if (diag_index < sizeA + sizeB) {
        uint mp = merge_pointer_full(a, b, c, sizeA, sizeB, diag_index);
        c[diag_index] = mp >= sizeA ? b[mp - sizeA] : a[mp];
    }
}







__kernel void new_merge_full(__global const uint *a,
                            __global const uint *b,
                            __global uint *c,
                            uint sizeA,
                            uint sizeB) {
    merge_global_full(a, b, c, sizeA, sizeB);
}


