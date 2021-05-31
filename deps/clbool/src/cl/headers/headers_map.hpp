#include <unordered_map>
#include <merge_path.h>
#include <merge_path1d.h>
#include <dcsr_addition_count.h>
#include <hash___hash_global.h>
#include <submatrix.h>
#include <reduce.h>
#include <hash___bitonic_sort.h>
#include <hash___hash_global.h>
#include <hash___hash_pwarp.h>
#include <hash___hash_tb.h>
#include <prepare_positions.h>
#include <coo_bitonic_sort.h>
#include <set_positions.h>
#include <prefix_sum.h>
#include <coo_kronecker.h>
#include <dscr_to_coo.h>
#include <for_test___half_sized_scan.h>
#include <to_result_matrix_single_thread.h>
#include <to_result_matrix_work_group.h>
#include <heap_merge.h>
#include <copy_one_value.h>
#include <merge_large_rows.h>
#include <bitonic_esc.h>
#include <count_workload.h>
#include <for_test___new_merge.h>
#include <dcsr_kronecker.h>
#include <coo_reduce_duplicates.h>
struct KernelSource {
    const char* kernel;
    size_t length;
};
static const std::unordered_map<std::string, KernelSource> HeadersMap = {
        {"merge_path", {merge_path_kernel, merge_path_kernel_length}},
        {"merge_path1d", {merge_path1d_kernel, merge_path1d_kernel_length}},
        {"dcsr_addition_count", {dcsr_addition_count_kernel, dcsr_addition_count_kernel_length}},
        {"hash/hash_global", {hash___hash_global_kernel, hash___hash_global_kernel_length}},
        {"submatrix", {submatrix_kernel, submatrix_kernel_length}},
        {"reduce", {reduce_kernel, reduce_kernel_length}},
        {"hash/bitonic_sort", {hash___bitonic_sort_kernel, hash___bitonic_sort_kernel_length}},
        {"hash/hash_global", {hash___hash_global_kernel, hash___hash_global_kernel_length}},
        {"hash/hash_pwarp", {hash___hash_pwarp_kernel, hash___hash_pwarp_kernel_length}},
        {"hash/hash_tb", {hash___hash_tb_kernel, hash___hash_tb_kernel_length}},
        {"prepare_positions", {prepare_positions_kernel, prepare_positions_kernel_length}},
        {"coo_bitonic_sort", {coo_bitonic_sort_kernel, coo_bitonic_sort_kernel_length}},
        {"set_positions", {set_positions_kernel, set_positions_kernel_length}},
        {"prefix_sum", {prefix_sum_kernel, prefix_sum_kernel_length}},
        {"coo_kronecker", {coo_kronecker_kernel, coo_kronecker_kernel_length}},
        {"dscr_to_coo", {dscr_to_coo_kernel, dscr_to_coo_kernel_length}},
        {"for_test/half_sized_scan", {for_test___half_sized_scan_kernel, for_test___half_sized_scan_kernel_length}},
        {"to_result_matrix_single_thread", {to_result_matrix_single_thread_kernel, to_result_matrix_single_thread_kernel_length}},
        {"to_result_matrix_work_group", {to_result_matrix_work_group_kernel, to_result_matrix_work_group_kernel_length}},
        {"heap_merge", {heap_merge_kernel, heap_merge_kernel_length}},
        {"copy_one_value", {copy_one_value_kernel, copy_one_value_kernel_length}},
        {"merge_large_rows", {merge_large_rows_kernel, merge_large_rows_kernel_length}},
        {"bitonic_esc", {bitonic_esc_kernel, bitonic_esc_kernel_length}},
        {"count_workload", {count_workload_kernel, count_workload_kernel_length}},
        {"for_test/new_merge", {for_test___new_merge_kernel, for_test___new_merge_kernel_length}},
        {"dcsr_kronecker", {dcsr_kronecker_kernel, dcsr_kronecker_kernel_length}},
        {"coo_reduce_duplicates", {coo_reduce_duplicates_kernel, coo_reduce_duplicates_kernel_length}},
};
