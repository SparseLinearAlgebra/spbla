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

#ifndef SPBLA_TESTING_MATRIXPRINTING_HPP
#define SPBLA_TESTING_MATRIXPRINTING_HPP

#include <spbla/spbla.h>

namespace testing {

    template <typename Stream>
    void printMatrix(Stream& stream, const spbla_Index* rowsIndex, const spbla_Index* colsIndex, spbla_Index nrows, spbla_Index ncols, spbla_Index nvals) {
        spbla_Index currentRow = 0;
        spbla_Index currentCol = 0;
        spbla_Index currentId = 0;

        while (currentId < nvals) {
            auto i = rowsIndex[currentId];
            auto j = colsIndex[currentId];

            while (currentRow < i) {
                while (currentCol < ncols) {
                    stream << "." << " ";
                    currentCol += 1;
                }

                stream << "\n";
                currentRow += 1;
                currentCol = 0;
            }

            while (currentCol < j) {
                stream << "." << " ";
                currentCol += 1;
            }

            stream << "1" << " ";
            currentId += 1;
            currentCol += 1;
        }

        while (currentRow < nrows) {
            while (currentCol < ncols) {
                stream << "." << " ";
                currentCol += 1;
            }

            stream << "\n";
            currentRow += 1;
            currentCol = 0;
        }
    }

    struct Print {
        spbla_Matrix matrix;
    };

    template <typename Stream>
    Stream& operator <<(Stream& stream, Print p) {
        spbla_Matrix matrix = p.matrix;
        assert(matrix);

        spbla_Index nrows;
        spbla_Index ncols;
        spbla_Index nvals;

        // Query matrix data
        spbla_Matrix_Nrows(matrix, &nrows);
        spbla_Matrix_Ncols(matrix, &ncols);
        spbla_Matrix_Nvals(matrix, &nvals);

        std::vector<spbla_Index> rowIndex(nvals);
        std::vector<spbla_Index> colIndex(nvals);

        spbla_Matrix_ExtractPairs(matrix, rowIndex.data(), colIndex.data(), &nvals);
        printMatrix(stream, rowIndex.data(), colIndex.data(), nrows, ncols, nvals);

        return stream;
    }

    template <typename Stream>
    Stream& operator <<(Stream& stream, const Matrix& a) {
        printMatrix(stream, a.rowsIndex.data(), a.colsIndex.data(), a.nrows, a.ncols, a.nvals);
        return stream;
    }
    
}

#endif //SPBLA_TESTING_MATRIXPRINTING_HPP
