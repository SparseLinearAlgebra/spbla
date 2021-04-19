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

// Fills sparse matrix with random data and tests whether the transfer works correctly
void testMatrixSetElement(spbla_Index m, spbla_Index n, float density) {
    spbla_Matrix matrix = nullptr;

    testing::Matrix tmatrix = std::move(testing::Matrix::generateSparse(m, n, density));

    auto I = tmatrix.rowsIndex;
    auto J = tmatrix.colsIndex;
    size_t nvals = tmatrix.nvals;
    size_t dups = nvals * 0.10;

    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

    if (nvals > 0) {
        for (size_t k = 0; k < dups; k++) {
            size_t id = std::min<size_t>(dist(engine) * nvals, nvals - 1);
            auto i = I[id], j = J[id];
            I.push_back(i);
            J.push_back(j);
        }
    }

    ASSERT_EQ(spbla_Matrix_New(&matrix, m, n), SPBLA_STATUS_SUCCESS);

    nvals = I.size();
    for (size_t i = 0; i < nvals; i++) {
        ASSERT_EQ(spbla_Matrix_SetElement(matrix, I[i], J[i]), SPBLA_STATUS_SUCCESS);
    }

    // Compare test matrix and library one
    ASSERT_TRUE(tmatrix.areEqual(matrix));

    // Remember to release resources
    ASSERT_EQ(spbla_Matrix_Free(matrix), SPBLA_STATUS_SUCCESS);
}

void testMatrixAppendElement(spbla_Index m, spbla_Index n, float density) {
    spbla_Matrix matrix = nullptr;

    testing::Matrix tmatrix = std::move(testing::Matrix::generateSparse(m, n, density));

    ASSERT_EQ(spbla_Matrix_New(&matrix, m, n), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Build(matrix, tmatrix.rowsIndex.data(), tmatrix.colsIndex.data(), (spbla_Index) tmatrix.nvals, SPBLA_HINT_VALUES_SORTED | SPBLA_HINT_NO_DUPLICATES), SPBLA_STATUS_SUCCESS);

    std::vector<spbla_Index> I;
    std::vector<spbla_Index> J;
    size_t nvals = tmatrix.nvals;
    size_t dups = nvals * 0.20;

    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

    if (nvals > 0) {
        for (size_t k = 0; k < dups; k++) {
            size_t id = std::min<size_t>(dist(engine) * nvals, nvals - 1);
            auto i = tmatrix.rowsIndex[id], j = tmatrix.colsIndex[id];
            I.push_back(i);
            J.push_back(j);
        }
    }

    nvals = I.size();
    for (size_t i = 0; i < nvals; i++) {
        ASSERT_EQ(spbla_Matrix_SetElement(matrix, I[i], J[i]), SPBLA_STATUS_SUCCESS);
    }

    // Compare test matrix and library one
    ASSERT_TRUE(tmatrix.areEqual(matrix));

    // Remember to release resources
    ASSERT_EQ(spbla_Matrix_Free(matrix), SPBLA_STATUS_SUCCESS);
}

void testMatrixPostAppendElement(spbla_Index m, spbla_Index n, float density) {
    spbla_Matrix matrix = nullptr;
    spbla_Matrix duplicated = nullptr;

    testing::Matrix tmatrix = std::move(testing::Matrix::generateSparse(m, n, density));

    ASSERT_EQ(spbla_Matrix_New(&matrix, m, n), SPBLA_STATUS_SUCCESS);

    for (size_t i = 0; i < tmatrix.nvals; i++) {
        ASSERT_EQ(spbla_Matrix_SetElement(matrix, tmatrix.rowsIndex[i], tmatrix.colsIndex[i]), SPBLA_STATUS_SUCCESS);
    }

    ASSERT_EQ(spbla_Matrix_Duplicate(matrix, &duplicated), SPBLA_STATUS_SUCCESS);

    // Compare test matrix and library one
    ASSERT_TRUE(tmatrix.areEqual(duplicated));

    // Remember to release resources
    ASSERT_EQ(spbla_Matrix_Free(matrix), SPBLA_STATUS_SUCCESS);
    ASSERT_EQ(spbla_Matrix_Free(duplicated), SPBLA_STATUS_SUCCESS);
}

void testRun(spbla_Index m, spbla_Index n, spbla_Hints setup) {
    ASSERT_EQ(spbla_Initialize(setup), SPBLA_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixSetElement(m, n, 0.001f + (0.05f) * ((float) i));
    }

    for (size_t i = 0; i < 10; i++) {
        testMatrixAppendElement(m, n, 0.001f + (0.05f) * ((float) i));
    }

    for (size_t i = 0; i < 10; i++) {
        testMatrixPostAppendElement(m, n, 0.001f + (0.05f) * ((float) i));
    }

    ASSERT_EQ(spbla_Finalize(), SPBLA_STATUS_SUCCESS);
}

#ifdef SPBLA_WITH_CUDA
TEST(spbla_Matrix, SetElementSmallCuda) {
    spbla_Index m = 60, n = 100;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, SetElementMediumCuda) {
    spbla_Index m = 500, n = 1000;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}

TEST(spbla_Matrix, SetElementLargeCuda) {
    spbla_Index m = 1000, n = 2000;
    testRun(m, n, SPBLA_HINT_CUDA_BACKEND);
}
#endif

#ifdef SPBLA_WITH_OPENCL
TEST(spbla_Matrix, SetElementSmallOpenCL) {
    spbla_Index m = 60, n = 100;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, SetElementMediumOpenCL) {
    spbla_Index m = 500, n = 1000;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}

TEST(spbla_Matrix, SetElementLargeOpenCL) {
    spbla_Index m = 1000, n = 2000;
    testRun(m, n, SPBLA_HINT_OPENCL_BACKEND);
}
#endif

#ifdef SPBLA_WITH_SEQUENTIAL
TEST(spbla_Matrix, SetElementSmallFallback) {
    spbla_Index m = 60, n = 100;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, SetElementMediumFallback) {
    spbla_Index m = 500, n = 1000;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}

TEST(spbla_Matrix, SetElementLargeFallback) {
    spbla_Index m = 1000, n = 2000;
    testRun(m, n, SPBLA_HINT_CPU_BACKEND);
}
#endif

SPBLA_GTEST_MAIN