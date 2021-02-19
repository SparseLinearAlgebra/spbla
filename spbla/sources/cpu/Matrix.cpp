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

#include <cpu/Matrix.hpp>
#include <vector>
#include <cassert>

namespace spbla {
    namespace cpu {

        Matrix::Matrix(size_t nrows, size_t ncols) {
            GrB_Info info = GrB_Matrix_new(&mMatrix, GrB_BOOL, (GrB_Index) nrows, (GrB_Index) ncols);

            if (info != GrB_Info::GrB_SUCCESS) {

            }
        }

        Matrix::~Matrix() {
            if (mMatrix) {
                GrB_Matrix_free(&mMatrix);
                mMatrix = nullptr;
            }
        }

        void Matrix::Build(const index *rowIds, const index *colIds, size_t nvals, bool isSorted) {
            assert(rowIds);
            assert(colIds);

            // Convert input data into appropriate GrB format
            std::vector<GrB_Index> i(nvals);
            std::vector<GrB_Index> j(nvals);
            bool* x = (bool*) std::malloc(nvals * sizeof(bool));

            for (size_t id = 0; id < nvals; id++) {
                i[id] = rowIds[id];
                j[id] = colIds[id];
                x[id] = true;
            }

            GrB_Info info = GrB_Matrix_build_BOOL(mMatrix, i.data(), j.data(), x, (GrB_Index) nvals, GrB_FIRST_BOOL);

            // Release unmanaged resource first
            std::free(x);

            if (info != GrB_Info::GrB_SUCCESS) {

            }
        }

        void Matrix::Extract(index *rowIds, index *colIds, size_t &nvals) const {
            assert(rowIds);
            assert(colIds);

            GrB_Index actualNvals = GetNvals();

            if (actualNvals > nvals) {

            }

            // Read data into tmp buffer, then fill the user provided arrays
            std::vector<GrB_Index> i(actualNvals);
            std::vector<GrB_Index> j(actualNvals);
            bool* x = (bool*) std::malloc(actualNvals * sizeof(bool));

            GrB_Matrix_extractTuples_BOOL(i.data(), j.data(), x, &actualNvals, mMatrix);

            size_t next = 0;

            for (size_t id = 0; id < actualNvals; id++) {
                if (x[id]) {
                    rowIds[next] = i[id];
                    colIds[next] = j[id];
                    next += 1;
                }
            }

            // Release unmanaged resource first
            std::free(x);

            assert(actualNvals == next && "Expect that GrB does not store false values");

            // Assign the actual number of nnz values
            nvals = actualNvals;
        }

        void Matrix::Multiply(const backend::Matrix &matrixA, const backend::Matrix &matrixB) {
            const auto* a = dynamic_cast<const Matrix*>(&matrixA);
            const auto* b = dynamic_cast<const Matrix*>(&matrixB);

            if (!a || !b) {

            }

            assert(a->GetNcols() == b->GetNrows());

            auto grbA = a->mMatrix;
            auto grbB = b->mMatrix;

            GrB_Info info = GrB_mxm(mMatrix, nullptr, nullptr, GrB_LOR_LAND_SEMIRING_BOOL, grbA, grbB, nullptr);

            if (info != GrB_Info::GrB_SUCCESS) {

            }
        }

        void Matrix::EWiseAdd(const backend::Matrix &matrixA, const backend::Matrix &matrixB) {
            const auto* a = dynamic_cast<const Matrix*>(&matrixA);
            const auto* b = dynamic_cast<const Matrix*>(&matrixB);

            if (!a || !b) {

            }

            assert(a->GetNrows() == b->GetNrows());
            assert(a->GetNcols() == b->GetNcols());

            auto grbA = a->mMatrix;
            auto grbB = b->mMatrix;

            GrB_Info info = GrB_Matrix_eWiseAdd_BinaryOp(mMatrix, nullptr, nullptr, GrB_LOR, grbA, grbB, nullptr);

            if (info != GrB_Info::GrB_SUCCESS) {

            }
        }

        void Matrix::Kronecker(const backend::Matrix &matrixA, const backend::Matrix &matrixB) {
            const auto* a = dynamic_cast<const Matrix*>(&matrixA);
            const auto* b = dynamic_cast<const Matrix*>(&matrixB);

            if (!a || !b) {

            }

            auto grbA = a->mMatrix;
            auto grbB = b->mMatrix;

            GrB_Info info = GrB_Matrix_kronecker_BinaryOp(mMatrix, nullptr, nullptr, GrB_LAND, grbA, grbB, nullptr);

            if (info != GrB_Info::GrB_SUCCESS) {

            }
        }

        size_t Matrix::GetNrows() const {
            GrB_Index nrows = 0;
            GrB_Info info = GrB_Matrix_nrows(&nrows, mMatrix);

            if (info != GrB_Info::GrB_SUCCESS) {

            }

            return (size_t) nrows;
        }

        size_t Matrix::GetNcols() const {
            GrB_Index ncols = 0;
            GrB_Info info = GrB_Matrix_ncols(&ncols, mMatrix);

            if (info != GrB_Info::GrB_SUCCESS) {

            }

            return (size_t) ncols;
        }

        size_t Matrix::GetNvals() const {
            GrB_Index nvals = 0;
            GrB_Info info = GrB_Matrix_nvals(&nvals, mMatrix);

            if (info != GrB_Info::GrB_SUCCESS) {

            }

            return (size_t) nvals;
        }
    }
}
