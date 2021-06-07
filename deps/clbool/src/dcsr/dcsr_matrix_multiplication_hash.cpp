#include <numeric>
#include <sstream>
#include "dcsr_matrix_multiplication.hpp"
#include "dcsr_matrix_multiplication_hash.hpp"

const uint32_t BINS_NUM = 8;
const uint32_t MAX_GROUP_ID = BINS_NUM - 1;
#define PWARP 4
namespace clbool::dcsr {
    
    namespace hash_details {

        uint32_t get_block_size(uint32_t bin_id) {
            // NOTE: NVIDIA can operate more than 256 threads per group, but AMD cannot
            if (bin_id == 0) return 256;
            if (bin_id == 1) return 64;
            if (bin_id == 2) return 128;
            if (bin_id == 3) return 256;
            if (bin_id == 4) return 256;
            if (bin_id == 5) return 256;
            if (bin_id == 6) return 256;
            if (bin_id == 7) return 256;
            CLB_RAISE("Invalid bin_id: " + std::to_string(bin_id) + ". Possible values from 0 to 7 inc.",
                            CLBOOL_INVALID_ARGUMENT);
        }

        uint32_t get_group(uint32_t size) {
            if (size <= 32) return 0;
            if (size <= 128) return 1;
            if (size <= 256) return 2;
            if (size <= 512) return 3;
            if (size <= 1024) return 4;
            if (size <= 2048) return 5;
            if (size <= 4096) return 6;
            return 7;
        }

        uint32_t get_table_size(uint32_t bin_id) {
            if (bin_id == 1) return 128;
            if (bin_id == 2) return 256;
            if (bin_id == 3) return 512;
            if (bin_id == 4) return 1024;
            if (bin_id == 5) return 2048;
            if (bin_id == 6) return 4096;
            CLB_RAISE("Invalid bin_id: " + std::to_string(bin_id) + ". Possible values from 1 to 6 inc.",
                            CLBOOL_INVALID_ARGUMENT);
        }
    }


    void matrix_multiplication_hash(Controls &controls,
                                    matrix_dcsr &matrix_out,
                                    const matrix_dcsr &a,
                                    const matrix_dcsr &b) {

        if (a.ncols() != b.nrows()) {
            std::stringstream s;
            s << "Invalid input matrix size! a : " << a.nrows() << " x " << a.ncols()
            << ", b: " << b.nrows() << " x " << b.ncols();
            CLB_RAISE(s.str(), CLBOOL_INVALID_ARGUMENT);
        }

        
        if (a.nnz() == 0 || b.nnz() == 0) {
            matrix_out = matrix_dcsr(a.nrows(), b.ncols());
            return;
        }

        cl::Buffer nnz_estimation;

        {
            START_TIMING
            count_workload(controls, nnz_estimation, a, b);
            END_TIMING("count_workload: ")
        }

        std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
        cpu_buffer groups_pointers(BINS_NUM + 1);
        cpu_buffer groups_length(BINS_NUM);


        cl::Buffer global_hash_tables;
        cl::Buffer global_hash_tables_offset;
        uint32_t pre_nnz;
        {
            START_TIMING
            build_groups_and_allocate_hash(controls, pre_nnz, cpu_workload_groups, nnz_estimation, a,
                                           global_hash_tables, global_hash_tables_offset);
            END_TIMING("build_groups_and_allocate_hash: ")
        }
        if (pre_nnz == 0) {
            matrix_out = matrix_dcsr(a.nrows(), b.ncols());
            return;
        }

        cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());

        {
            START_TIMING
            write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);
            END_TIMING("write_bins_info: ")
        }

        {
            START_TIMING
            count_nnz(controls, groups_length, groups_pointers, gpu_workload_groups, nnz_estimation,
                      a, b, global_hash_tables, global_hash_tables_offset);
            END_TIMING("count_nnz: ")
        }


        {
            START_TIMING
            fill_nnz(controls, groups_length, groups_pointers, gpu_workload_groups, nnz_estimation,
                     matrix_out, a, b, global_hash_tables, global_hash_tables_offset);
            END_TIMING("fill_nnz: ")
        }

    }


    void count_nnz(Controls &controls,
                   const cpu_buffer &groups_length,
                   const cpu_buffer &groups_pointers,

                   const cl::Buffer &gpu_workload_groups,
                   cl::Buffer &nnz_estimation,

                   const matrix_dcsr &a,
                   const matrix_dcsr &b,

                   cl::Buffer &global_hash_tables,
                   const cl::Buffer &global_hash_tables_offset

    ) {
        auto hash_pwarp = kernel<cl::Buffer, uint32_t, uint32_t, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("hash/hash_pwarp", "hash_symbolic_pwarp");
        hash_pwarp.set_async(true);
        auto hash_tb = kernel<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("hash/hash_tb", "hash_symbolic_tb");
        hash_tb.set_async(true);
        auto hash_global = kernel<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t, cl::Buffer, cl::Buffer>("hash/hash_global", "hash_symbolic_global");
        hash_global.set_async(true);

        std::vector<cl::Event> events;
        for (uint32_t bin_id = 0; bin_id < BINS_NUM; ++bin_id) {
            if (groups_length[bin_id] == 0) continue;

            uint32_t block_size = hash_details::get_block_size(bin_id);

            LOG << "\n[count_nnz] group " << bin_id << ", size " << groups_length[bin_id];

            if (bin_id == 0) {
                hash_pwarp.set_work_size(groups_length[bin_id] * PWARP);
                cl::Event ev;
                CLB_RUN(
                ev = hash_pwarp.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
                                       nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
                                       b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                       b.nzr()));

                events.push_back(ev);
                continue;
            }

            if (bin_id != MAX_GROUP_ID) {

                hash_tb.set_block_size(block_size);
                hash_tb.add_option("TABLE_SIZE", hash_details::get_table_size(bin_id));
                hash_tb.set_work_size(block_size * groups_length[bin_id]);

                cl::Event ev;
                CLB_RUN(
                ev = hash_tb.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
                                             nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
                                             b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                             b.nzr()))

                events.push_back(ev);
                continue;
            }

            hash_global.set_block_size(block_size);
            hash_global.set_work_size(block_size * groups_length[bin_id]);

            cl::Event ev;
            CLB_RUN(
            ev = hash_global.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
                                             nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
                                             b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                             b.nzr(),
                                             global_hash_tables, global_hash_tables_offset))
            events.push_back(ev);
        }

        CLB_WAIT(cl::Event::waitForEvents(events));
    }

    void fill_nnz(Controls &controls,
                  const cpu_buffer &groups_length,
                  const cpu_buffer &groups_pointers,

                  const cl::Buffer &gpu_workload_groups,
                  cl::Buffer &pre_matrix_rows_pointers,

                  matrix_dcsr &c,
                  const matrix_dcsr &a,
                  const matrix_dcsr &b,

                  const cl::Buffer &global_hash_tables,
                  cl::Buffer &global_hash_tables_offset
    ) {
        uint32_t c_nnz;
        prefix_sum(controls, pre_matrix_rows_pointers, c_nnz, a.nzr() + 1);
        cl::Buffer c_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nnz);

        auto hash_pwarp = kernel<cl::Buffer, uint32_t, uint32_t, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("hash/hash_pwarp", "hash_numeric_pwarp");
        hash_pwarp.set_async(true);
        auto hash_tb = kernel<cl::Buffer, uint32_t, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("hash/hash_tb", "hash_numeric_tb");
        hash_tb.set_async(true);
        auto hash_global = kernel<cl::Buffer, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                ("hash/hash_global", "hash_numeric_global");
        hash_global.set_async(true);

        std::vector<cl::Event> events;
        for (uint32_t bin_id = 0; bin_id < BINS_NUM; ++bin_id) {
            if (groups_length[bin_id] == 0) continue;

            uint32_t block_size = hash_details::get_block_size(bin_id);
            if (bin_id == 0) {
                hash_pwarp.set_work_size(groups_length[bin_id] * PWARP);
                cl::Event ev;
                CLB_RUN(
                ev = hash_pwarp.run(controls, gpu_workload_groups, groups_pointers[bin_id], groups_length[bin_id],
                                       pre_matrix_rows_pointers, c_cols, a.rpt_gpu(), a.cols_gpu(),
                                       b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                       b.nzr()));

                events.push_back(ev);
                continue;
            }

            if (bin_id != MAX_GROUP_ID) {

                hash_tb.set_block_size(block_size);
                hash_tb.add_option("TABLE_SIZE", hash_details::get_table_size(bin_id));
                hash_tb.set_work_size(block_size * groups_length[bin_id]);

                cl::Event ev;
                CLB_RUN(
                ev = hash_tb.run(controls, gpu_workload_groups, groups_pointers[bin_id],
                                             pre_matrix_rows_pointers, c_cols, a.rpt_gpu(), a.cols_gpu(),
                                             b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                             b.nzr()));

                events.push_back(ev);

                continue;
            }

            hash_global.set_block_size(block_size);
            hash_global.set_work_size(block_size * groups_length[bin_id]);

            cl::Event ev;
            CLB_RUN(ev = hash_global.run(controls, gpu_workload_groups, groups_pointers[bin_id],
                                             pre_matrix_rows_pointers, c_cols,
                                             global_hash_tables, global_hash_tables_offset));
            events.push_back(ev);
        }

        CLB_WAIT(cl::Event::waitForEvents(events));

        cl::Buffer positions;
        CLB_CREATE_BUF(positions = utils::create_buffer(controls, a.nzr() + 1));

        prepare_positions(controls, positions, pre_matrix_rows_pointers, a.nzr(), "prepare_for_shift_empty_rows");

        uint32_t c_nzr;
        prefix_sum(controls, positions, c_nzr, a.nzr() + 1);

        cl::Buffer c_rpt;
        CLB_CREATE_BUF(c_rpt =  utils::create_buffer(controls, c_nzr + 1));

        cl::Buffer c_rows;
        CLB_CREATE_BUF(c_rows = utils::create_buffer(controls, c_nzr));

        set_positions(controls, c_rpt, c_rows, pre_matrix_rows_pointers, a.rows_gpu(), positions, a.nzr());

        c = matrix_dcsr(c_rpt, c_rows, c_cols, a.nrows(), b.ncols(), c_nnz, c_nzr);
    }

    void build_groups_and_allocate_hash(Controls &controls,
                                        uint32_t &pre_nnz_out,
                                        std::vector<cpu_buffer> &cpu_workload_groups,
                                        cl::Buffer &nnz_estimation,
                                        const matrix_dcsr &a,

                                        cl::Buffer &global_hash_tables,
                                        cl::Buffer &global_hash_tables_offset
    ) {

        cpu_buffer global_hash_tables_offset_cpu;
        uint32_t global_hash_mem_size = 0;

        cpu_buffer cpu_workload(a.nzr());

        CLB_READ_BUF(controls.queue.enqueueReadBuffer(nnz_estimation, CL_TRUE, 0, sizeof(uint32_t) * a.nzr(), cpu_workload.data()));

        uint32_t pre_nnz;
        for (uint32_t i = 0; i < a.nzr(); ++i) {
            uint32_t current_workload = cpu_workload[i];
            uint32_t group = hash_details::get_group(current_workload);
            cpu_workload_groups[group].push_back(i);

            pre_nnz += current_workload;
            if (group == MAX_GROUP_ID) {
                global_hash_tables_offset_cpu.push_back(global_hash_mem_size);
                global_hash_mem_size += current_workload;
            }
        }
        pre_nnz_out = pre_nnz;
        if (pre_nnz == 0) {
            return;
        }

        global_hash_tables_offset_cpu.push_back(global_hash_mem_size);

        if (global_hash_mem_size != 0) {

            CLB_CREATE_BUF(
                    global_hash_tables_offset = utils::create_buffer(controls, global_hash_tables_offset_cpu, true));

            CLB_CREATE_BUF (global_hash_tables = utils::create_buffer(controls, global_hash_mem_size));
        }

    }

}