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
#include <cassert>


namespace spbla {

    OpenCLMatrix::OpenCLMatrix(size_t nrows, size_t ncols)
    : mNrows(nrows)
    , mNcols(ncols)
    , mNvals(0)
    {}

    OpenCLMatrix::~OpenCLMatrix() {
        //RAISE_ERROR(NotImplemented, "This function must be implemented");
    }

    void OpenCLMatrix::setElement(index i, index j) {
        RAISE_ERROR(NotImplemented, "This function is not supported for this matrix class");
    }

    void OpenCLMatrix::clone(const MatrixBase &otherBase) {
        auto other = dynamic_cast<const OpenCLMatrix*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to csr matrix class");
        CHECK_RAISE_ERROR(other != this, InvalidArgument, "Matrices must differ");

        size_t M = other->getNrows();
        size_t N = other->getNcols();

        assert(this->getNrows() == M);
        assert(this->getNcols() == N);

        this->mMatrixImpl = other->mMatrixImpl;
    }

    index OpenCLMatrix::getNrows() const {
        return mNrows;
    }

    index OpenCLMatrix::getNcols() const {
        return mNcols;
    }

    index OpenCLMatrix::getNvals() const {
        return mNvals;
    }
}