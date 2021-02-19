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

#ifndef SPBLA_CPU_BACKEND_HPP
#define SPBLA_CPU_BACKEND_HPP

#include <backend/Backend.hpp>
#include <backend/Matrix.hpp>

namespace spbla {
    namespace cpu {

        class Backend final: public backend::Backend {
        public:
            ~Backend() override = default;

            void Initialize(uint32_t argc, const char *const *argv) override;
            bool IsInitialized() const override;
            void Finalize() override;

            backend::Matrix *CreateMatrix(size_t nrows, size_t ncols) override;
            void ReleaseMatrix(backend::Matrix *matrix) override;

            const std::string &GetName() const override;
            const std::string &GetDescription() const override;
            const std::string &GetAuthorsName() const override;

        private:
            bool mIsInitialized = false;
        };

    }
}

#endif //SPBLA_CPU_BACKEND_HPP