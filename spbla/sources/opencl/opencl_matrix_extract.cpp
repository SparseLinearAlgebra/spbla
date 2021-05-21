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

#include <opencl/opencl_matrix.hpp>
#include <core/error.hpp>
#include <matrix_coo.hpp>
#include <utils.hpp>
#include <matrices_conversions.hpp>

namespace spbla {

    void OpenCLMatrix::extract(index *rows, index *cols, size_t &nvals) {
        cl::Event evRow, evCol;
        clbool::matrix_coo mCoo = clbool::dcsr_to_coo_shallow(*clboolState, mMatrixImpl);

        clboolState->queue.enqueueReadBuffer(mCoo.rows_gpu(), false, 0, sizeof(index) * nvals,
                                              rows, nullptr, &evRow);
        clboolState->queue.enqueueReadBuffer(mCoo.cols_gpu(), false, 0, sizeof(index) * nvals,
                                              cols, nullptr, &evCol);
        evRow.wait(); evCol.wait();

        updateFromImpl();
    }

}