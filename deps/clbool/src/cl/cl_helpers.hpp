#pragma once
#include "clion_defines.cl"

inline void print_local_array(__local uint *arr, uint size) {
    for (uint i = 0; i < size; ++i) {
        printf("(%d, %d), ", i, arr[i]);
    }
    printf("\n");
}

inline void print_global_array(__global uint *arr, uint size) {
    for (uint i = 0; i < size; ++i) {
        printf("(%d, %d), ", i, arr[i]);
    }
    printf("\n");
}

inline void check_global_array(__global const uint *arr, uint size) {
    printf("start global check\n");
    for (uint i = 1; i < size; ++i) {
        if (arr[i] <= arr[i - 1]) {
            printf("oooops i = %d, i - 1 = %d\n", i, i - 1);
        };
    }
    printf("end global check\n");
    printf("\n");
}

inline void check_local_array(__local const uint *arr, uint size) {
    printf("start local check\n");
    for (uint i = 1; i < size; ++i) {
        if (arr[i] <= arr[i - 1]) {
            printf("oooops i = %d, i - 1 = %d\n", i, i - 1);
        };
    }
    printf("end local check\n");
    printf("\n");
}

