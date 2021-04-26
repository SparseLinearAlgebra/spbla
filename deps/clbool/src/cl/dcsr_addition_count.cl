#ifndef RUN

#include "clion_defines.cl"
#define GROUP_SIZE 256

#endif

const uint MAX_VAL = 4294967295;

__kernel void merge_path_count(const uint* a_rows_pointers,
                               const uint* a_cols,
                               const uint* b_rows_pointers,
                               const uint* b_cols,
                               const uint* c_rows_length) {

    const uint row = get_global_id(0);

    const T global_offset_a = a_rows_pointers[row];
    const T sz_a = a_rows_pointers[row + 1] - global_offset_a;

    const T global_offset_b = b_rows_pointers[row];
    const T sz_b = b_rows_pointers[row + 1] - global_offset_b;

    const T block_count = (sz_a + sz_b + block_size - 1) / block_size;

    T begin_a = 0;
    T begin_b = 0;

    __shared__ T raw_a[block_size + 2];
    __shared__ T raw_b[block_size + 2];
    __shared__ T res[block_size];

    bool dir = true;
    T item_from_prev_chank = MAX_VAL;

    for (auto i = 0; i < block_count; i++) {
        __shared__ T max_x_index;
        __shared__ T max_y_index;

        T max_x_index_per_thread = 0;
        T max_y_index_per_thread = 0;

        assert(sz_a >= begin_a);
        assert(sz_b >= begin_b);

        T buf_a_size = min(sz_a - begin_a, block_size);
        T buf_b_size = min(sz_b - begin_b, block_size);

        if (threadIdx.x == 0) {
            max_x_index = 0;
            max_y_index = 0;
        }

        for (auto j = threadIdx.x; j < block_size + 2; j += blockDim.x) {
            if (j > 0 && j - 1 < buf_a_size) {
                raw_a[j] = a_cols[global_offset_a + j - 1 + begin_a];
            } else {
                raw_a[j] = MAX_VAL;
            }
            if (j > 0 && j - 1 < buf_b_size) {
                raw_b[j] = b_cols[global_offset_b + j - 1 + begin_b];
            } else {
                raw_b[j] = MAX_VAL;
            }
        }

        __syncthreads();

        const T to_process = min(buf_b_size + buf_a_size, block_size);

        for (auto j = threadIdx.x; j < to_process; j += blockDim.x) {
            const T y = j + 2;
            const T x = 0;

            T l = 0;
            T r = j + 2;

            while (r - l > 1) {
                bool ans = raw_b[y - l - (r - l) / 2] > raw_a[x + l + (r - l) / 2];

                l += (r - l) / 2 * ans;
                r -= (r - l) / 2 * !ans;
            }

            T ans_x = x + l;
            T ans_y = y - l;

            if (ans_y == 1 || ans_x == 0) {
                if (ans_y == 1) {
                    res[j] = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                } else {
                    res[j] = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                }
            } else {
                if (raw_b[ans_y - 1] > raw_a[ans_x]) {
                    res[j] = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                } else {
                    res[j] = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                }
            }
        }

        atomicMax(&max_x_index, max_x_index_per_thread);
        atomicMax(&max_y_index, max_y_index_per_thread);

        __syncthreads();

        T counter = 0;

        if (dir) {
            for (auto m = threadIdx.x; m < to_process; m += blockDim.x) {
                if (m > 0)
                    counter += (res[m] - res[m - 1]) != 0;
                else
                    counter += (res[0] - item_from_prev_chank) != 0;
                item_from_prev_chank = res[m];
            }
        } else {
            for (auto m = blockDim.x - 1 - threadIdx.x; m < to_process; m += blockDim.x) {
                if (m > 0)
                    counter += (res[m] - res[m - 1]) != 0;
                else
                    counter += (res[0] - item_from_prev_chank) != 0;
                item_from_prev_chank = res[m];
            }
        }

        dir = !dir;

        atomicAdd(c_rows_length.get() + row, counter);

        begin_a += max_x_index;
        begin_b += max_y_index;

        __syncthreads();
    }
}