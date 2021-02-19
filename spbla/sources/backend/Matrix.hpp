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

#ifndef SPBLA_MATRIX_HPP
#define SPBLA_MATRIX_HPP

#include <core/Defines.hpp>
#include <memory>

namespace spbla {
    namespace backend {

        /**
         * Generic matrix type. Must be implemented by the backend.
         */
        class Matrix {
        public:
            virtual ~Matrix() = default;

            virtual void Build(const index* rowIds, const index* colIds, size_t nvals, bool isSorted) = 0;
            virtual void Extract(index* rowIds, index* colIds, size_t& nvals) const = 0;

            virtual void Multiply(const Matrix& matrixA, const Matrix& matrixB) = 0;
            virtual void EWiseAdd(const Matrix& matrixA, const Matrix& matrixB) = 0;
            virtual void Kronecker(const Matrix& matrixA, const Matrix& matrixB) = 0;

            virtual size_t GetNrows() const = 0;
            virtual size_t GetNcols() const = 0;
            virtual size_t GetNvals() const = 0;
        };

    }
}

#endif //SPBLA_MATRIX_HPP
