#pragma once

#include <cstdint>

namespace clbool {


    namespace details {

        class matrix_base {
        public:
            using index_type = uint32_t;
        protected:
            index_type _nrows;
            index_type _ncols;
            index_type _nnz = 0;

        public:

            matrix_base()
            : _nrows(0), _ncols(0), _nnz(0)
            {}

            matrix_base(index_type n_rows, index_type n_cols, index_type n_entities)
            : _nrows(n_rows), _ncols(n_cols), _nnz(n_entities)
            {}

            index_type nrows() const {
                return _nrows;
            };

            index_type ncols() const {
                return _ncols;
            };

            index_type nnz() const {
                return _nnz;
            };


            index_type nrows() {
                return _nrows;
            };

            index_type ncols() {
                return _ncols;
            };

            index_type nnz() {
                return _nnz;
            };

            bool empty() const {
                return _nnz == 0;
            }
        };
    }
}