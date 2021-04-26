#pragma once

#include "matrix_base.hpp"
#include "controls.hpp"

#include <vector>


namespace clbool {

    class matrix_dcsr : public details::matrix_base {

    private:
        // buffers for uint32only;
        cl::Buffer _rpt_gpu;
        cl::Buffer _rows_gpu;
        cl::Buffer _cols_gpu;

        uint32_t _nzr = 0;

    public:

        // -------------------------------------- constructors -----------------------------

        matrix_dcsr() = default;


        matrix_dcsr(cl::Buffer rpt_gpu,
                    cl::Buffer rows_gpu,
                    cl::Buffer cols_gpu,

                    uint32_t n_rows,
                    uint32_t n_cols,
                    uint32_t nnz,
                    uint32_t nzr
                    )
        : details::matrix_base(n_rows, n_cols, nnz)
        , _rpt_gpu(std::move(rpt_gpu))
        , _rows_gpu(std::move(rows_gpu))
        , _cols_gpu(std::move(cols_gpu))
        , _nzr(nzr)
        {};


        const auto &rpt_gpu() const {
            return _rpt_gpu;
        }

        const auto &rows_gpu() const {
            return _rows_gpu;
        }

        const auto &cols_gpu() const {
            return _cols_gpu;
        }

        const uint32_t &nzr() const {
            return _nzr;
        }

        auto &rpt_gpu()  {
            return _rpt_gpu;
        }

        auto &rows_gpu()  {
            return _rows_gpu;
        }

        auto &cols_gpu() {
            return _cols_gpu;
        }

        uint32_t &nzr()  {
            return _nzr;
        }

    };
}
