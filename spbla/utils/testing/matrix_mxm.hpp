/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020, 2021 JetBrains-Research                                    */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#ifndef SPBLA_TESTING_MATRIXMXM_HPP
#define SPBLA_TESTING_MATRIXMXM_HPP

#include <testing/matrix.hpp>
#include <testing/matrix_ewiseadd.hpp>
#include <algorithm>
#include <limits>

namespace testing {

    struct MatrixMultiplyFunctor {
        Matrix operator()(const Matrix& a, const Matrix& b, const Matrix& c, bool accumulate) {
            Matrix out;
            out.nrows = a.nrows;
            out.ncols = b.ncols;

            assert(a.ncols == b.nrows);
            assert(a.nrows == c.nrows);
            assert(b.ncols == c.ncols);

            a.computeRowOffsets();
            b.computeRowOffsets();

            spbla_Index max = std::numeric_limits<spbla_Index>::max();

            // Evaluate total nnz and nnz per row
            size_t nvals = 0;
            out.rowOffsets.resize(a.nrows + 1);
            std::vector<spbla_Index> mask(b.ncols, max);

            for (spbla_Index i = 0; i < a.nrows; i++) {
                size_t nvalsInRow = 0;

                for (spbla_Index ak = a.rowOffsets[i]; ak < a.rowOffsets[i + 1]; ak++) {
                    spbla_Index k = a.colsIndex[ak];

                    for (spbla_Index bk = b.rowOffsets[k]; bk < b.rowOffsets[k + 1]; bk++) {
                        spbla_Index j = b.colsIndex[bk];

                        // Do not compute col nnz twice
                        if (mask[j] != i) {
                            mask[j] = i;
                            nvalsInRow += 1;
                        }
                    }
                }

                nvals += nvalsInRow;
                out.rowOffsets[i] = nvalsInRow;
            }

            // Row offsets
            spbla_Index sum = 0;
            for (auto& rowOffset: out.rowOffsets) {
                spbla_Index next = sum + rowOffset;
                rowOffset = sum;
                sum = next;
            }
            out.hasRowOffsets = true;

            out.nvals = nvals;
            out.rowsIndex.resize(nvals);
            out.colsIndex.resize(nvals);

            // Fill column indices per row and sort
            mask.clear();
            mask.resize(b.ncols, max);

            for (spbla_Index i = 0; i < a.nrows; i++) {
                size_t id = 0;
                size_t first = out.rowOffsets[i];
                size_t last = out.rowOffsets[i + 1];

                for (spbla_Index ak = a.rowOffsets[i]; ak < a.rowOffsets[i + 1]; ak++) {
                    spbla_Index k = a.colsIndex[ak];

                    for (spbla_Index bk = b.rowOffsets[k]; bk < b.rowOffsets[k + 1]; bk++) {
                        spbla_Index j = b.colsIndex[bk];

                        // Do not compute col nnz twice
                        if (mask[j] != i) {
                            mask[j] = i;
                            out.rowsIndex[first + id] = i;
                            out.colsIndex[first + id] = j;
                            id += 1;
                        }
                    }
                }

                // Sort indices within row
                std::sort(out.colsIndex.begin() + first, out.colsIndex.begin() + last);
            }

            if (accumulate) {
                MatrixEWiseAddFunctor functor;
                return functor(c, out);
            }

            return std::move(out);
        }
    };

}

#endif //SPBLA_TESTING_MATRIXMXM_HPP
