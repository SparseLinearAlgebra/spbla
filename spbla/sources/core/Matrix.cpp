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

#include <core/Matrix.hpp>
#include <cassert>

namespace spbla {

    Matrix::Matrix(size_t nrows, size_t ncols, backend::Backend& provider) {
        mHnd = provider.CreateMatrix(nrows, ncols);
        mProvider = &provider;
    }

    Matrix::~Matrix() {
        if (mHnd) {
            mProvider->ReleaseMatrix(mHnd);
            mHnd = nullptr;
        }
    }

    void Matrix::Validate() const {
        assert(mMagicNumber == MAGIC_NUMBER);
        assert(mHnd != nullptr);
        assert(mProvider != nullptr);
    }

    void Matrix::Build(const index *rowIds, const index *colIds, size_t nvals, bool isSorted) {
        // todo: validation of the state and arguments

        mHnd->Build(rowIds, colIds, nvals, isSorted);
    }

    void Matrix::Extract(index *rowIds, index *colIds, size_t &nvals) const {
        // todo: validation of the state and arguments

        mHnd->Extract(rowIds, colIds, nvals);
    }

    void Matrix::Multiply(const backend::Matrix &matrixA, const backend::Matrix &matrixB) {
        const auto* a = dynamic_cast<const Matrix*>(&matrixA);
        const auto* b = dynamic_cast<const Matrix*>(&matrixB);

        // todo: validation of the state and arguments

        mHnd->Multiply(*a->mHnd, *b->mHnd);
    }

    void Matrix::EWiseAdd(const backend::Matrix &matrixA, const backend::Matrix &matrixB) {
        const auto* a = dynamic_cast<const Matrix*>(&matrixA);
        const auto* b = dynamic_cast<const Matrix*>(&matrixB);

        // todo: validation of the state and arguments

        mHnd->EWiseAdd(*a->mHnd, *b->mHnd);
    }

    void Matrix::Kronecker(const backend::Matrix &matrixA, const backend::Matrix &matrixB) {
        const auto* a = dynamic_cast<const Matrix*>(&matrixA);
        const auto* b = dynamic_cast<const Matrix*>(&matrixB);

        // todo: validation of the state and arguments

        mHnd->Kronecker(*a->mHnd, *b->mHnd);
    }

    size_t Matrix::GetNrows() const {
        return mHnd->GetNrows();
    }

    size_t Matrix::GetNcols() const {
        return mHnd->GetNcols();
    }

    size_t Matrix::GetNvals() const {
        return mHnd->GetNvals();
    }

}