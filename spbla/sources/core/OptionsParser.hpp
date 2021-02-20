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

#ifndef SPBLA_OPTIONSPARSER_HPP
#define SPBLA_OPTIONSPARSER_HPP

#include <core/Defines.hpp>
#include <unordered_map>
#include <string>
#include <cassert>

namespace spbla {

    /**
     * Parses input options for backends setup.
     */
    class OptionsParser {
    public:
        OptionsParser() = default;
        ~OptionsParser() = default;

        void Parse(index optionsCount, const char* const* options) {
            index i = 0;

            while (i < optionsCount) {
                std::string option = options[i];

                if (option.rfind("--", 0) == 0) {
                    std::string name = option.substr(2);
                    mOptions.emplace(std::move(name), "");
                    i += 1;
                    continue;
                }

                if (option.rfind("-", 0) == 0) {
                    std::string name = option.substr(1);
                    i += 1;

                    if (i < optionsCount) {
                        std::string arg = options[i];
                        mOptions.emplace(std::move(name), std::move(arg));
                        i += 1;
                        continue;
                    }

                    return;
                }
            }

            mParsed = true;
        }

        bool Has(const std::string& option) const {
            auto found = mOptions.find(option);
            return found != mOptions.end();
        }

        const std::string& Get(const std::string& option) const {
            auto found = mOptions.find(option);
            assert(found != mOptions.end());
            return found->second;
        }

        bool IsParsed() const {
            return mParsed;
        }

    private:
        bool mParsed = false;
        std::unordered_map<std::string, std::string> mOptions;
    };

}

#endif //SPBLA_OPTIONSPARSER_HPP
