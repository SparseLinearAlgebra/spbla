/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2021 JetBrains-Research                                          */
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

#ifndef SPBLA_TESTING_MATRIX_HPP
#define SPBLA_TESTING_MATRIX_HPP

#include <spbla/spbla.h>
#include <testing/Pair.hpp>
#include <testing/MatrixGenerator.hpp>
#include <algorithm>
#include <chrono>
#include <cassert>

namespace testing {

    // Default sizes
    const spbla_Index SMALL_M = 50, SMALL_N = 60;
    const spbla_Index MEDIUM_M = 500, MEDIUM_N = 600;
    const spbla_Index LARGE_M = 1500, LARGE_N = 2600;

    struct Matrix {
        std::vector<spbla_Index> mRowsIndex;
        std::vector<spbla_Index> mColsIndex;
        size_t mNvals = 0;
        size_t mNrows = 0;
        size_t mNcols = 0;

        bool AreEqual(spbla_Matrix matrix) const {
            spbla_Index extNvals;

            EXPECT_EQ(spbla_Matrix_Nvals(matrix, &extNvals), SPBLA_INFO_SUCCESS);

            std::vector<spbla_Index> extRows(extNvals);
            std::vector<spbla_Index> extCols(extNvals);

            EXPECT_EQ(spbla_Matrix_Extract(matrix, extRows.data(), extCols.data(), &extNvals, SPBLA_HINT_NO), SPBLA_INFO_SUCCESS);

            if (extNvals != mNvals)
                return false;

            std::vector<Pair> extracted(mNvals);
            std::vector<Pair> reference(mNvals);

            for (spbla_Index idx = 0; idx < mNvals; idx++) {
                extracted[idx] = Pair{extRows[idx], extCols[idx]};
                reference[idx] = Pair{mRowsIndex[idx], mColsIndex[idx]};
            }

            std::sort(extracted.begin(), extracted.end(), PairCmp());
            std::sort(reference.begin(), reference.end(), PairCmp());

            return extracted == reference;
        }

        template<typename Condition>
        static Matrix Generate(size_t nrows, size_t ncols, Condition&& condition) {
            Matrix matrix;
            matrix.mNrows = nrows;
            matrix.mNcols = ncols;
            GenerateTestData(nrows, ncols, matrix.mRowsIndex, matrix.mColsIndex, matrix.mNvals, std::forward<Condition>(condition));
            return matrix;
        }

        static Matrix GenerateSparse(size_t nrows, size_t ncols, double density) {
            Matrix matrix;
            matrix.mNrows = nrows;
            matrix.mNcols = ncols;

            std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());

            size_t totalVals = nrows * ncols;
            size_t valsToGen = totalVals * density;

            auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

            std::unordered_set<Pair,PairHash,PairEq> indices;

            for (size_t id = 0; id < valsToGen; id++) {
                auto pr = dist(engine);
                auto pc = dist(engine);

                auto r = (spbla_Index) (pr * (float) nrows);
                auto c = (spbla_Index) (pc * (float) ncols);

                r = std::min(r, (spbla_Index)nrows - 1);
                c = std::min(c, (spbla_Index)ncols - 1);

                indices.emplace(Pair{r, c});
            }

            matrix.mNvals = indices.size();
            matrix.mRowsIndex.reserve(matrix.mNvals);
            matrix.mColsIndex.reserve(matrix.mNvals);

            for (auto& p: indices) {
                matrix.mRowsIndex.push_back(p.i);
                matrix.mColsIndex.push_back(p.j);
            }

            return std::move(matrix);
        }
    };
    
}

#endif //SPBLA_TESTING_MATRIX_HPP