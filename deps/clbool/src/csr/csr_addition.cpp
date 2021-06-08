#include "csr.hpp"

#include <sstream>

namespace clbool::csr {
    uint32_t NUM_BINS = 4;

    uint32_t get_bin_id(uint32_t row_size) {
        if (row_size == 0) return 0;
        if (row_size <= 64) return 1;
        if (row_size <= 128) return 2;
        return 3;
    }

    uint32_t get_block_size(uint32_t bin_id) {
        if (bin_id == 0) return 128;
        if (bin_id == 1) return 64;
        if (bin_id == 2) return 128;
        if (bin_id == 3) return  256;
        std::stringstream s;
        s << "Invalid bin id " << bin_id << ", possible values: 1--3.";
        CLB_RAISE(s.str(), CLBOOL_INVALID_ARGUMENT);
    }


    void matrix_addition(Controls &controls, matrix_csr &c, const matrix_csr &a, const matrix_csr &b) {

        if (a.nrows() != b.nrows() || a.ncols() != b.ncols()) {
            std::stringstream s;
            s << "Invalid matrixes size! a: " << a.nrows() << " x " << a.ncols() <<
              ", b: " << b.nrows() << " x " << b.ncols();
            CLB_RAISE(s.str(), CLBOOL_INVALID_ARGUMENT);
        }

        if (a.empty() && b.empty()) {
            c = matrix_csr(a.nrows(), a.ncols());
            return;
        }

        if (a.empty() || b.empty()) {
            const matrix_csr &empty = a.empty() ? a : b;
            const matrix_csr &filled = a.empty() ? b : a;

            if (&c == &filled) return;

            cl::Buffer rpt;
            cl::Buffer cols;
            CLB_CREATE_BUF(rpt = utils::create_buffer(controls, filled.nrows() + 1));
            CLB_CREATE_BUF(cols = utils::create_buffer(controls, filled.nnz()));

            CLB_COPY_BUF(controls.queue.enqueueCopyBuffer(filled.rpt_gpu(), rpt, 0, 0, sizeof(uint32_t) * (filled.nrows() + 1)));
            CLB_COPY_BUF(controls.queue.enqueueCopyBuffer(filled.cols_gpu(), cols, 0, 0, sizeof(uint32_t) * filled.nnz()));
            c = matrix_csr(rpt, cols, filled.nrows(), filled.ncols(), filled.nnz());
            return;
        }

        // ---------------------------------- estimate load -----------------------------------
        cl::Buffer c_rpt;
        cl::Buffer permutation;
        cl::Buffer bins_offset;
        cpu_buffer bins_offset_cpu(NUM_BINS + 1, 0);
        cl::Buffer bins_size;
        CLB_CREATE_BUF(permutation = utils::create_buffer(controls, a.nrows()));
        CLB_CREATE_BUF(bins_offset = utils::create_buffer(controls, NUM_BINS + 1));
        CLB_CREATE_BUF(bins_size = utils::create_buffer(controls, NUM_BINS));
        CLB_CREATE_BUF(c_rpt = utils::create_buffer(controls, a.nrows() + 1));

        // init all
        {
            auto init = kernel<cl::Buffer, uint32_t>
                    ("csr_addition", "init_with_zeroes");
            init.set_work_size(a.nrows() + 1);
            init.set_block_size(controls.max_wg_size);
            CLB_RUN(init.run(controls, c_rpt, a.nrows() + 1));

            init.set_work_size(NUM_BINS + 1);
            CLB_RUN(init.run(controls, bins_offset, NUM_BINS + 1));

            init.set_work_size(NUM_BINS);
            CLB_RUN(init.run(controls, bins_size, NUM_BINS));
        }

        {
            START_TIMING
            auto fill_bins_size = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                    ("csr_addition", "fill_bins_size");
            fill_bins_size.set_block_size(controls.max_wg_size);
            fill_bins_size.set_work_size(a.nrows());
            CLB_RUN(fill_bins_size.run(controls, a.rpt_gpu(), b.rpt_gpu(), bins_offset, a.nrows()));
            END_TIMING("bills filled in: ")
        }
        uint32_t total;
        prefix_sum(controls, bins_offset, total, NUM_BINS + 1);

        {
            START_TIMING
            auto build_permutation = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                    ("csr_addition", "build_permutation");
            build_permutation.set_block_size(controls.max_wg_size);
            build_permutation.set_work_size(a.nrows());
            CLB_RUN(build_permutation.run(controls, a.rpt_gpu(), b.rpt_gpu(), bins_offset, bins_size, permutation, a.nrows()));
        }

        CLB_READ_BUF(controls.queue.enqueueReadBuffer(bins_offset, true, 0, sizeof(uint32_t) * (NUM_BINS + 1),
                                                      bins_offset_cpu.data()))

        {
            START_TIMING
            std::vector<cl::Event> events;
            auto add_symbolic = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                                uint32_t, cl::Buffer, uint32_t>
                                    ("csr_addition", "addition_symbolic");
            add_symbolic.set_async(true);

            // 0 cat be ignored
            for (uint32_t i = 1; i < NUM_BINS; ++i) {
                uint32_t bin_size = bins_offset_cpu[i + 1] - bins_offset_cpu[i];
                if (bin_size == 0) continue;
                uint32_t bs = get_block_size(i);
                add_symbolic.set_block_size(bs);
                add_symbolic.set_work_size(bs * bin_size);
                cl::Event ev;
                CLB_RUN(ev = add_symbolic.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(), c_rpt, a.nrows(),
                                         permutation, bins_offset_cpu[i]));
                events.push_back(ev);
            }

            CLB_WAIT(cl::WaitForEvents(events));
            END_TIMING("symbolic part run in: ")
        }

        uint32_t c_nnz;

        {
            START_TIMING
            prefix_sum(controls, c_rpt, c_nnz, a.nrows() + 1);
            END_TIMING("prefix_sum run in: ")
        }

        cl::Buffer c_cols;
        CLB_CREATE_BUF(c_cols = utils::create_buffer(controls, c_nnz));

        {
            START_TIMING
            std::vector<cl::Event> events;
            auto add_numeric = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                                                            cl::Buffer, uint32_t>
                        ("csr_addition", "addition_numeric");
            add_numeric.set_async(true);

            for (uint32_t i = 1; i < NUM_BINS; ++i) {
                uint32_t bin_size = bins_offset_cpu[i + 1] - bins_offset_cpu[i];
                if (bin_size == 0) continue;
                uint32_t bs = get_block_size(i);
                add_numeric.set_block_size(bs);
                add_numeric.set_work_size(bs * bin_size);
                cl::Event ev;
                CLB_RUN(ev = add_numeric.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(), c_rpt, c_cols,
                                        permutation, bins_offset_cpu[i]));
                events.push_back(ev);
            }

            CLB_WAIT(cl::WaitForEvents(events));
            END_TIMING("numeric part run in: ")
        }

        c = matrix_csr(c_rpt, c_cols, a.nrows(), a.ncols(), c_nnz);
    }
}

