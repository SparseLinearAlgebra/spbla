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

void testMatrixKronecker(spbla_Index m, spbla_Index n, spbla_Index k, spbla_Index t, float density, spbla_Hints flags) {
    spbla_Matrix r, a, b;

    testing::Matrix ta = std::move(testing::Matrix::generateSparse(m, n, density));
    testing::Matrix tb = std::move(testing::Matrix::generateSparse(k, t, density));

    // Allocate input matrices and resize to fill with input data
    EXPECT_EQ(spbla_Matrix_New(&a, m, n), SPBLA_STATUS_SUCCESS);
    EXPECT_EQ(spbla_Matrix_New(&b, k, t), SPBLA_STATUS_SUCCESS);
    EXPECT_EQ(spbla_Matrix_New(&r, m * k, n * t), SPBLA_STATUS_SUCCESS);

    // Transfer input data into input matrices
    EXPECT_EQ(spbla_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, SPBLA_HINT_VALUES_SORTED), SPBLA_STATUS_SUCCESS);
    EXPECT_EQ(spbla_Matrix_Build(b, tb.rowsIndex.data(), tb.colsIndex.data(), tb.nvals, SPBLA_HINT_VALUES_SORTED), SPBLA_STATUS_SUCCESS);

    // Evaluate r = a `kron` b
    EXPECT_EQ(spbla_Kronecker(r, a, b, flags), SPBLA_STATUS_SUCCESS);

    // Evaluate naive r = a `kron` b on the cpu to compare results
    testing::MatrixKroneckerFunctor functor;
    testing::Matrix tr = std::move(functor(ta, tb));

    // Compare results
    EXPECT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    EXPECT_EQ(spbla_Matrix_Free(a), SPBLA_STATUS_SUCCESS);
    EXPECT_EQ(spbla_Matrix_Free(b), SPBLA_STATUS_SUCCESS);
    EXPECT_EQ(spbla_Matrix_Free(r), SPBLA_STATUS_SUCCESS);
}

void testRun(spbla_Index m, spbla_Index n, spbla_Index k, spbla_Index t, float step, spbla_Hints setup) {
    // Setup library
    EXPECT_EQ(spbla_Initialize(setup), SPBLA_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixKronecker(m, n, k, t, 0.01f + step * ((float) i), SPBLA_HINT_NO);
    }

    // Finalize library
    EXPECT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

TEST(spbla_Matrix, KroneckerSmall) {
    spbla_Index m = 10, n = 20;
    spbla_Index k = 5, t = 15;
    float step = 0.05f;
    testRun(m, n, k, t, step, SPBLA_HINT_NO);
}

TEST(spbla_Matrix, KroneckerMedium) {
    spbla_Index m = 100, n = 40;
    spbla_Index k = 30, t = 80;
    float step = 0.02f;
    testRun(m, n, k, t, step, SPBLA_HINT_NO);
}

TEST(spbla_Matrix, KroneckerLarge) {
    spbla_Index m = 1000, n = 400;
    spbla_Index k = 300, t = 800;
    float step = 0.001f;
    testRun(m, n, k, t, step, SPBLA_HINT_NO);
}

TEST(spbla_Matrix, KroneckerSmallFallback) {
    spbla_Index m = 10, n = 20;
    spbla_Index k = 5, t = 15;
    float step = 0.05f;
    testRun(m, n, k, t, step, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, KroneckerMediumFallback) {
    spbla_Index m = 100, n = 40;
    spbla_Index k = 30, t = 80;
    float step = 0.02f;
    testRun(m, n, k, t, step, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, KroneckerLargeFallback) {
    spbla_Index m = 1000, n = 400;
    spbla_Index k = 300, t = 800;
    float step = 0.001f;
    testRun(m, n, k, t, step, SPBLA_HINT_CPU_BACKEND);
}

SPBLA_GTEST_MAIN