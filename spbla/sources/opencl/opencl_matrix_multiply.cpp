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

namespace spbla {

    void OpenCLMatrix::multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate, bool checkTime) {
        CHECK_RAISE_ERROR(clboolState != nullptr, InvalidState, "Clbool state isn't initialized!")

        auto a = dynamic_cast<const clbool::matrix_dcsr*>(&aBase);
        auto b = dynamic_cast<const clbool::matrix_dcsr*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to clbool::matrix_dcsr class")
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to clbool::matrix_dcsr class")

        clbool::dcsr::matrix_multiplication_hash(*clboolState, mMatrixImpl, *a, *b);
        updateFromImpl();
    }

}