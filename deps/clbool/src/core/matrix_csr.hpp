#pragma once

#include "matrix_base.hpp"
#include "controls.hpp"
namespace clbool {

    class matrix_csr : public details::matrix_base {
    private:
        cl::Buffer _rpt_gpu;
        cl::Buffer _cols_gpu;

    public:

        matrix_csr(Controls &controls, uint32_t nrows, uint32_t ncols, uint32_t nvals)
                : matrix_base(nrows, ncols, nvals)
        {}

        matrix_csr() = default;

        matrix_csr(cl::Buffer rpt_gpu,
                   cl::Buffer cols_gpu,
                   uint32_t nrows,
                   uint32_t ncols,
                   uint32_t nvals)
                : matrix_base(nrows, ncols, nvals)
                , _rpt_gpu(std::move(rpt_gpu))
                , _cols_gpu(std::move(cols_gpu))
        {}

        matrix_csr(uint32_t nrows, uint32_t ncols)
        : details::matrix_base(nrows, ncols, 0)
        {}

        const auto& rpt_gpu() const {
            return _rpt_gpu;
        }

        const auto& cols_gpu() const {
            return _cols_gpu;
        }

    };
}