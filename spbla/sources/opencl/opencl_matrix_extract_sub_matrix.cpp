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
#include <dcsr/dcsr.hpp>
#include <cassert>

namespace spbla {

    void OpenCLMatrix::extractSubMatrix(const MatrixBase &otherBase, index i, index j, index nrows, index ncols, bool checkTime) {
        assert(this->getNrows() == nrows);
        assert(this->getNcols() == ncols);

        auto otherDcsr = dynamic_cast<const OpenCLMatrix*>(&otherBase);

        CHECK_RAISE_ERROR(otherDcsr != nullptr, InvalidArgument, "Passed matrix does not belong to clbool::matrix_dcsr class");

        clbool::dcsr::submatrix(*clboolState, mMatrixImpl, otherDcsr->mMatrixImpl, i, j, nrows, ncols);
        updateFromImpl();
    }
}