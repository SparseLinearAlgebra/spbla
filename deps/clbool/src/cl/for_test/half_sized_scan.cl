#ifndef RUN
#include "../clion_defines.cl"
#define GROUP_SIZE 256
#endif
//// count prefixes itself on input array
//// and vertices as thr output

// TODO: optimise bank conflicts
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
__kernel void scan_blelloch_half(
        __global unsigned int * pref_sum,
        unsigned int n)
{

    __local uint tmp[GROUP_SIZE * 2];
    uint local_id = get_local_id(0);
    uint local_id_second_half = local_id + GROUP_SIZE;
    uint block_size = get_local_size(0);
    uint doubled_block_size = get_local_size(0) * 2;

    uint dp = 1;
    tmp[local_id] = local_id < n ? pref_sum[local_id] : 0;
    tmp[local_id_second_half] = local_id_second_half < n ? pref_sum[local_id_second_half] : 0;

    for(uint s = doubled_block_size>>1; s > 1; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;
            tmp[j] += tmp[i];
        }

        if(local_id_second_half < s)
        {
            uint i = dp*(2 * local_id_second_half + 1) - 1;
            uint j = dp*(2 * local_id_second_half + 2) - 1;
            tmp[j] += tmp[i];
        }

        dp <<= 1;
    }

    tmp[doubled_block_size - 1] += tmp[block_size - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        printf("total sum: %d\n", tmp[doubled_block_size - 1]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id == block_size - 1) {
        unsigned int t = tmp[local_id];
        tmp[local_id] = 0;
        tmp[local_id_second_half] = t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        printf("half: %d\n", tmp[doubled_block_size - 1]);
    }

    for(uint s = 2; s < doubled_block_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;

            unsigned int t = tmp[j];
            tmp[j] += tmp[i];
            tmp[i] = t;
        }

        if(local_id_second_half < s)
        {
            uint i = dp*(2 * local_id_second_half + 1) - 1;
            uint j = dp*(2 * local_id_second_half + 2) - 1;

            unsigned int t = tmp[j];
            tmp[j] += tmp[i];
            tmp[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < n) {
        pref_sum[local_id] = tmp[local_id];
    }

    if (local_id_second_half < n) {
        pref_sum[local_id_second_half] = tmp[local_id_second_half];
    }
}
