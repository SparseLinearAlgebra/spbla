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

#include <testing/testing.hpp>

void testMatrixMultiplyAdd(spbla_Index m, spbla_Index t, spbla_Index n, float density, spbla_Hints flags) {
    spbla_Matrix a, b, r;

    // Generate test data with specified density
    testing::Matrix ta = std::move(testing::Matrix::generateSparse(m, t, density));
    testing::Matrix tb = std::move(testing::Matrix::generateSparse(t, n, density));
    testing::Matrix tr = std::move(testing::Matrix::generateSparse(m, n, density));

    // Allocate input matrices and resize to fill with input data
    ASSERT_EQ(spbla_Matrix_New(&a, m, t), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_New(&b, t, n), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_New(&r, m, n), SPBLA_STATUS_SUCCESS);

    // Transfer input data into input matrices
    ASSERT_EQ(spbla_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, SPBLA_HINT_VALUES_SORTED & SPBLA_HINT_NO_DUPLICATES), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Build(b, tb.rowsIndex.data(), tb.colsIndex.data(), tb.nvals, SPBLA_HINT_VALUES_SORTED & SPBLA_HINT_NO_DUPLICATES), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Build(r, tr.rowsIndex.data(), tr.colsIndex.data(), tr.nvals, SPBLA_HINT_VALUES_SORTED & SPBLA_HINT_NO_DUPLICATES), SPBLA_STATUS_SUCCESS);

    // Evaluate naive r += a x b on the cpu to compare results
    testing::MatrixMultiplyFunctor functor;
    tr = std::move(functor(ta, tb, tr, true));

    // Evaluate r += a x b
    ASSERT_EQ(spbla_MxM(r, a, b, SPBLA_HINT_ACCUMULATE | flags), SPBLA_STATUS_SUCCESS);

    // Compare results
    ASSERT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    ASSERT_EQ(spbla_Matrix_Free(a), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(b), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(r), SPBLA_STATUS_SUCCESS);
}

void testMatrixMultiply(spbla_Index m, spbla_Index t, spbla_Index n, float density, spbla_Hints flags) {
    spbla_Matrix a, b, r;

    // Generate test data with specified density
    testing::Matrix ta = testing::Matrix::generateSparse(m, t, density);
    testing::Matrix tb = testing::Matrix::generateSparse(t, n, density);
    testing::Matrix tr = testing::Matrix::empty(m, n);

    // Allocate input matrices and resize to fill with input data
    ASSERT_EQ(spbla_Matrix_New(&a, m, t), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_New(&b, t, n), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_New(&r, m, n), SPBLA_STATUS_SUCCESS);

    // Transfer input data into input matrices
    ASSERT_EQ(spbla_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, SPBLA_HINT_VALUES_SORTED & SPBLA_HINT_NO_DUPLICATES), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Build(b, tb.rowsIndex.data(), tb.colsIndex.data(), tb.nvals, SPBLA_HINT_VALUES_SORTED & SPBLA_HINT_NO_DUPLICATES), SPBLA_STATUS_SUCCESS);

    // Evaluate naive r += a x b on the cpu to compare results
    testing::MatrixMultiplyFunctor functor;
    tr = std::move(functor(ta, tb, tr, false));

    // Evaluate r += a x b
    ASSERT_EQ(spbla_MxM(r, a, b, flags), SPBLA_STATUS_SUCCESS);

    // Compare results
    ASSERT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    ASSERT_EQ(spbla_Matrix_Free(a), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(b), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(r), SPBLA_STATUS_SUCCESS);
}

void testRun(spbla_Index m, spbla_Index t, spbla_Index n, spbla_Hints setup) {
    // Setup library
    ASSERT_EQ(spbla_Initialize(setup), SPBLA_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixMultiplyAdd(m, t, n, 0.1f + (0.05f) * ((float) i), SPBLA_HINT_NO);
    }

    for (size_t i = 0; i < 5; i++) {
        testMatrixMultiply(m, t, n, 0.1f + (0.05f) * ((float) i), SPBLA_HINT_NO);
    }

    // Finalize library
    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

TEST(spbla_Matrix, MultiplySmall) {
    spbla_Index m = 60, t = 100, n = 80;
    testRun(m, t, n, SPBLA_HINT_NO);
}

TEST(spbla_Matrix, MultiplyMedium) {
    spbla_Index m = 500, t = 1000, n = 800;
    testRun(m, t, n, SPBLA_HINT_NO);
}

TEST(spbla_Matrix, MultiplyLarge) {
    spbla_Index m = 1000, t = 2000, n = 500;
    testRun(m, t, n, SPBLA_HINT_NO);
}

TEST(spbla_Matrix, MultiplySmallFallback) {
    spbla_Index m = 60, t = 100, n = 80;
    testRun(m, t, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, MultiplyMediumFallback) {
    spbla_Index m = 500, t = 1000, n = 800;
    testRun(m, t, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, MultiplyLargeFallback) {
    spbla_Index m = 1000, t = 2000, n = 500;
    testRun(m, t, n, SPBLA_HINT_CPU_BACKEND);
}

SPBLA_GTEST_MAIN