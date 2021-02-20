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

#include <testing/Testing.hpp>

// Fills sparse matrix with random data and tests whether the transfer works correctly
void TestMatrixBuildExtract(spbla_Index m, spbla_Index n, float density) {
    spbla_Matrix matrix = nullptr;

    testing::Matrix tmatrix = testing::Matrix::Generate(m, n, testing::Condition3(density));

    EXPECT_EQ(spbla_Matrix_New(&matrix, m, n), SPBLA_INFO_SUCCESS);
    EXPECT_EQ(spbla_Matrix_Build(matrix, tmatrix.mRowsIndex.data(), tmatrix.mColsIndex.data(), tmatrix.mNvals, SPBLA_HINT_VALUES_SORTED), SPBLA_INFO_SUCCESS);

    // Compare test matrix and library one
    EXPECT_EQ(tmatrix.AreEqual(matrix), true);

    // Remember to release resources
    EXPECT_EQ(spbla_Matrix_Free(&matrix), SPBLA_INFO_SUCCESS);
}

void TestRun(spbla_Index m, spbla_Index n, spbla_Backend backend) {
    const char* options[] = { "--CPU_IGNORE_FINALIZATION" };
    EXPECT_EQ(spbla_Initialize(backend, 1, options), SPBLA_INFO_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        TestMatrixBuildExtract(m, n, 0.001f + (0.05f) * ((float) i));
    }

    EXPECT_EQ(spbla_Finalize(), SPBLA_INFO_SUCCESS);
}

TEST(spbla_Matrix, BuildExtractSmall) {
    TestRun(testing::SMALL_M, testing::SMALL_N, SPBLA_BACKEND_DEFAULT);
}

TEST(spbla_Matrix, BuildExtractMedium) {
    TestRun(testing::MEDIUM_M, testing::MEDIUM_N, SPBLA_BACKEND_DEFAULT);
}

TEST(spbla_Matrix, BuildExtractLarge) {
    TestRun(testing::LARGE_M, testing::LARGE_N, SPBLA_BACKEND_DEFAULT);
}

SPBLA_GTEST_MAIN