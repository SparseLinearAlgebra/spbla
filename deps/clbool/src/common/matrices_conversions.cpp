#include "matrices_conversions.hpp"

namespace clbool {
    namespace {
    #define CONV_GROUP_SIZE 64

        void create_rows_pointers(Controls &controls,
                                  cl::Buffer &rows_pointers_out,
                                  cl::Buffer &rows_compressed_out,
                                  const cl::Buffer &rows,
                                  uint32_t size,
                                  uint32_t &nzr // non zero rows_gpu
        ) {

            cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

            auto prepare_positions = kernel<cl::Buffer, cl::Buffer, uint32_t>
                    ("prepare_positions", "prepare_array_for_rows_positions");
            prepare_positions.set_work_size(size);

            prepare_positions.run(controls, positions, rows, size);

            prefix_sum(controls, positions, nzr, size);

            cl::Buffer rows_pointers(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr + 1));
            cl::Buffer rows_compressed(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr);

            auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>
                    ("set_positions", "set_positions_rows");
            set_positions.set_kernel_name("set_positions_rows")
                    .set_work_size(size);

            set_positions.run(controls, rows_pointers, rows_compressed, rows, positions, size, nzr);

            rows_pointers_out = std::move(rows_pointers);
            rows_compressed_out = std::move(rows_compressed);
        }
    }

    /*
     * .cols array in output and input is the same
     */
    matrix_coo dcsr_to_coo_shallow(Controls &controls, matrix_dcsr &a) {
        if (a.empty()) {
            return matrix_coo(a.nrows(), a.ncols());
        }

        cl::Buffer c_rows(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nnz());

        auto dscr_to_coo = kernel<cl::Buffer, cl::Buffer, cl::Buffer>
                ("dscr_to_coo", "dscr_to_coo");
        dscr_to_coo.set_block_size(CONV_GROUP_SIZE);
        dscr_to_coo.set_work_size(a.nzr() * CONV_GROUP_SIZE);

        dscr_to_coo.run(controls, a.rpt_gpu(), a.rows_gpu(), c_rows);
        return matrix_coo(c_rows, a.cols_gpu(), a.nrows(), a.ncols(), a.nnz());
    }

    matrix_coo dcsr_to_coo_deep(Controls &controls, const matrix_dcsr &a) {
        if (a.empty()) {
            return matrix_coo(a.nrows(), a.ncols());
        }

        cl::Buffer c_rows(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nnz());
        cl::Buffer c_cols(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nnz());

        controls.queue.enqueueCopyBuffer(a.cols_gpu(), c_cols, 0, 0, sizeof(matrix_dcsr::index_type) * a.nnz());

        auto dscr_to_coo = kernel<cl::Buffer, cl::Buffer, cl::Buffer>
                ("dscr_to_coo", "dscr_to_coo");
        dscr_to_coo.set_block_size(CONV_GROUP_SIZE)
                .set_work_size(a.nzr() * CONV_GROUP_SIZE);

        dscr_to_coo.run(controls, a.rpt_gpu(), a.rows_gpu(), c_rows);
        return matrix_coo(c_rows, c_cols, a.nrows(), a.ncols(), a.nnz());
    }

    matrix_dcsr coo_to_dcsr_shallow(Controls &controls, const matrix_coo &a) {
        if (a.empty()) {
            return matrix_dcsr(a.nrows(), a.ncols());
        }
        cl::Buffer rpt;
        cl::Buffer rows;
        uint32_t nzr;
        if (a.nnz() != 0)
            create_rows_pointers(controls, rpt, rows, a.rows_gpu(), a.nnz(), nzr);

        return matrix_dcsr(rpt, rows, a.cols_gpu(),
                           a.nrows(), a.ncols(), a.nnz(), nzr
        );
    }

    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const matrix_dcsr_cpu &m, uint32_t size) {
        if (m.cols().empty()) {
            return matrix_dcsr(size, size);
        }

        cl::Buffer rows_pointers(controls.context,
                                 (const_cast<matrix_dcsr_cpu&>(m)).rpt().begin(), (const_cast<matrix_dcsr_cpu&>(m)).rpt().end(), false);
        cl::Buffer rows_compressed(controls.context,
                                   (const_cast<matrix_dcsr_cpu&>(m)).rows().begin(), (const_cast<matrix_dcsr_cpu&>(m)).rows().end(), false);
        cl::Buffer cols_indices(controls.context,
                                (const_cast<matrix_dcsr_cpu&>(m)).cols().begin(), (const_cast<matrix_dcsr_cpu&>(m)).cols().end(), false);

        return matrix_dcsr(rows_pointers, rows_compressed, cols_indices,
                           size, size, m.cols().size(), m.rows().size());

    }

    matrix_coo matrix_coo_from_cpu(Controls &controls, matrix_coo_cpu &m, uint32_t size) {
        if (m.cols().empty()) {
            return matrix_coo(size, size);
        }

        cl::Buffer rows_indices(controls.context, m.rows().begin(), m.rows().end(), false);
        cl::Buffer cols_indices(controls.context, m.cols().begin(), m.cols().end(), false);

        return matrix_coo(rows_indices, cols_indices, size, size, m.rows().size());
    }

    matrix_dcsr_cpu matrix_dcsr_from_gpu(Controls &controls, matrix_dcsr &m) {
        if (m.empty()) {
            return matrix_dcsr_cpu();
        }

        cpu_buffer rows_pointers(m.nzr() + 1);
        cpu_buffer rows_compressed(m.nzr());
        cpu_buffer cols_indices(m.nnz());

        controls.queue.enqueueReadBuffer(m.rpt_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * rows_pointers.size(), rows_pointers.data());
        controls.queue.enqueueReadBuffer(m.rows_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * rows_compressed.size(), rows_compressed.data());
        controls.queue.enqueueReadBuffer(m.cols_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * cols_indices.size(), cols_indices.data());

        return matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);

    }


    matrix_coo_cpu matrix_coo_from_gpu(Controls &controls, matrix_coo &m) {
        if (m.empty()) {
            return matrix_coo_cpu();
        }

        cpu_buffer rows_indices(m.nnz());
        cpu_buffer cols_indices(m.nnz());

        controls.queue.enqueueReadBuffer(m.rows_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * rows_indices.size(), rows_indices.data());
        controls.queue.enqueueReadBuffer(m.cols_gpu(), CL_TRUE, 0,
                                         sizeof(matrix_dcsr::index_type) * cols_indices.size(), cols_indices.data());

        return matrix_coo_cpu(rows_indices, cols_indices);
    }

    matrix_csr_cpu csr_cpu_from_pairs(const matrix_coo_cpu_pairs &mat, uint32_t m, uint32_t n) {
        cpu_buffer rpt(m + 1);
        cpu_buffer cols(mat.size());
        rpt[m] = cols.size();
        int ptr = 0;
        int j = 0;
        for (uint32_t i = 0; i < m; ++i) {
            rpt[i] = ptr;
            while (j < mat.size() && mat[j].first == i) {
                cols[j] = mat[j].second;
                ptr ++;
                j ++;
            }
        }
//        for (uint32_t i = 1020; i < 1030; ++i) {
//            std::cout << rpt[i] << ", ";
//        }
//        std::cout << std::endl;
        return matrix_csr_cpu(std::move(rpt), std::move(cols), m, n);
    }

    matrix_csr_cpu csr_cpu_from_coo_cpu(const matrix_coo_cpu &mat, uint32_t m, uint32_t n) {
        uint32_t nnz = mat.cols().size();
        cpu_buffer rpt(m + 1);
        cpu_buffer cols(nnz);
        rpt[m] = nnz;
        int ptr = 0;
        int j = 0;
        for (uint32_t i = 0; i < m; ++i) {
            rpt[i] = ptr;
            while (j < nnz && mat.rows()[j] == i) {
                cols[j] = mat.cols()[j];
                ptr ++;
                j ++;
            }
        }

        return matrix_csr_cpu(std::move(rpt), std::move(cols), m, n);
    }


    matrix_csr csr_from_cpu(Controls &controls, const matrix_csr_cpu &m) {
        if (m.cols().empty()) {
            return matrix_csr(m.nrows(), m.ncols());
        }
        cl::Buffer rpt;
        cl::Buffer cols;
        CLB_CREATE_BUF(rpt = utils::create_buffer(controls, const_cast<cpu_buffer&>(m.rpt())))
        CLB_CREATE_BUF(cols = utils::create_buffer(controls, const_cast<cpu_buffer&>(m.cols())))
        return matrix_csr(rpt, cols, m.nrows(), m.ncols(), m.cols().size());
    }

    // shallow for cols!
    matrix_dcsr csr_to_dcsr(Controls &controls, const matrix_csr &m) {
        if (m.empty()) {
            return matrix_dcsr(m.nrows(), m.ncols());
        }

        cl::Buffer positions;
        CLB_CREATE_BUF(positions = utils::create_buffer(controls, m.nrows() + 1));

        {
            auto prepare_pos = kernel<cl::Buffer, cl::Buffer, uint32_t>
                    ("prepare_positions", "prepare_for_shift_empty_rows");
            prepare_pos.set_block_size(controls.max_wg_size);
            prepare_pos.set_work_size(m.nrows());

            CLB_RUN(prepare_pos.run(controls, positions, m.rpt_gpu(), m.nrows()))
        }

        uint32_t c_nzr;
        prefix_sum(controls, positions, c_nzr, m.nrows() + 1);
        cl::Buffer c_rpt;
        cl::Buffer c_rows;
        CLB_CREATE_BUF(c_rpt = utils::create_buffer(controls, c_nzr + 1));
        CLB_CREATE_BUF(c_rows = utils::create_buffer(controls, c_nzr));

        {
            auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                    ("set_positions", "set_positions_pointers_and_rows_csr");
            set_positions.set_block_size(controls.max_wg_size);
            set_positions.set_work_size(m.nrows());
            CLB_RUN(set_positions.run(controls, c_rpt, c_rows, m.rpt_gpu(), positions, m.nrows()));
        }

        return matrix_dcsr(c_rpt, c_rows, m.cols_gpu(), m.nrows(), m.ncols(), m.nnz(), c_nzr);
    }

    // shallow for cols!
    matrix_csr dcsr_to_csr(Controls &controls, const matrix_dcsr &m) {
        if (m.empty()) {
            return matrix_csr(m.nrows(), m.ncols());
        }

        cl::Buffer c_rpt;
        CLB_CREATE_BUF(c_rpt = utils::create_buffer(controls, m.nrows() + 1));
        fill_with_zeroes(controls, c_rpt, m.nrows() + 1);

        {
            auto set_rsize = kernel<cl::Buffer, cl::Buffer, uint32_t, cl::Buffer>
                    ("conversions", "dcsr_to_csr_set_size");
            set_rsize.set_block_size(controls.max_wg_size);
            set_rsize.set_work_size(m.nzr());

            CLB_RUN(set_rsize.run(controls, m.rpt_gpu(), m.rows_gpu(), m.nzr(), c_rpt));
        }

        uint32_t c_nnz;
        prefix_sum(controls, c_rpt, c_nnz, m.nrows() + 1);

        CLB_CHECK(c_nnz == m.nnz(), "nnz in dcsr and csr matrices should be equal!", CLBOOL_INVALID_VALUE);

        return matrix_csr(c_rpt, m.cols_gpu(), m.nrows(), m.ncols(), c_nnz);
    }

    matrix_csr_cpu matrix_csr_from_gpu(Controls &controls, const matrix_csr &m) {
        if (m.empty()) {
            return matrix_csr_cpu(m.nrows(), m.ncols());
        }

        cpu_buffer rpt(m.nrows() + 1);
        cpu_buffer cols(m.nnz());

        CLB_READ_BUF(utils::read_buffer(controls, rpt, m.rpt_gpu()).wait())
        CLB_READ_BUF(utils::read_buffer(controls, cols, m.cols_gpu()).wait())

        return matrix_csr_cpu(rpt, cols, m.nrows(), m.ncols());
    }
}