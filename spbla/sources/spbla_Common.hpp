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

#ifndef SPBLA_SPBLA_COMMON_HPP
#define SPBLA_SPBLA_COMMON_HPP

#include <spbla/spbla.h>
#include <core/config.hpp>
#include <core/version.hpp>
#include <core/error.hpp>
#include <core/library.hpp>
#include <core/matrix.hpp>

// State validation
#define SPBLA_VALIDATE_LIBRARY                                                         \
    spbla::Library::validate();

// Arguments validation
#define SPBLA_ARG_NOT_NULL(arg)                                                        \
    CHECK_RAISE_ERROR(arg != nullptr, InvalidArgument, "Passed null argument")

#define SPBLA_BEGIN_BODY                                                               \
    try {

#define SPBLA_END_BODY }                                                               \
    catch (const spbla::Error& err) {                                                  \
         spbla::Library::handleError(err);                                             \
         return err.getStatus();                                                        \
    }                                                                                   \
    catch (const std::exception& exc) {                                                 \
         spbla::Library::handleError(exc);                                             \
         return SPBLA_STATUS_ERROR;                                                    \
    }                                                                                   \
    return spbla_Status::SPBLA_STATUS_SUCCESS;

#endif //SPBLA_SPBLA_COMMON_HPP
