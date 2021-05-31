#pragma once


#include "matrix_base.hpp"
#include "controls.hpp"
#include "../coo/coo_initialization.hpp"
#include "../common/utils.hpp"
#include <vector>

namespace clbool {

    class matrix_coo : public details::matrix_base {
    private:
        // buffers for uint32only;
        cl::Buffer _rows;
        cl::Buffer _cols;

    private:
        void reduce_duplicates(Controls& controls);
        void reduce_duplicates2(Controls &controls);

    public:

        // -------------------------------------- constructors -----------------------------

        matrix_coo() = default;

        matrix_coo(index_type nrows,
                   index_type ncols);

        matrix_coo(index_type nrows,
                   index_type ncols,
                   index_type nnz,
                   cl::Buffer &rows,
                   cl::Buffer &cols
        );

        matrix_coo(Controls &controls,
                   index_type nrows,
                   index_type ncols,
                   index_type nnz,
                   const index_type *rows_indices,
                   const index_type *cols_indices,
                   bool sorted = true,
                   bool noDuplicates = true
                   );

        matrix_coo(Controls &controls,
                   index_type nrows,
                   index_type ncols,
                   index_type nnz,
                   cl::Buffer &rows,
                   cl::Buffer &cols,
                   bool sorted = true,
                   bool noDuplicates = true
                   );

        matrix_coo(matrix_coo const &other) = default;

        matrix_coo(matrix_coo &&other) noexcept = default;

        matrix_coo &operator=(const matrix_coo &other);

        const auto &rows_gpu() const {
            return _rows;
        }

        const auto &cols_gpu() const {
            return _cols;
        }

        auto &rows_gpu() {
            return _rows;
        }

        auto &cols_gpu() {
            return _cols;
        }

    };
}

