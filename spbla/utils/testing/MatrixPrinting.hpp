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

#ifndef SPBLA_TESTING_MATRIXPRINTING_HPP
#define SPBLA_TESTING_MATRIXPRINTING_HPP

#include <spbla/spbla.h>

namespace testing {

    template <typename Stream>
    void PrintMatrix(Stream& stream, const spbla_Index* rowsIndex, const spbla_Index* colsIndex, spbla_Index nrows, spbla_Index ncols, spbla_Index nvals) {
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
    
}

#endif //SPBLA_TESTING_MATRIXPRINTING_HPP
