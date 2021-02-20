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

#ifndef SPBLA_EXCEPTION_HPP
#define SPBLA_EXCEPTION_HPP

#include <spbla/spbla.h>
#include <exception>
#include <string>

namespace spbla {

    /**
     * Generic library exception.
     * Use this one in particular backend implementations.
     */
    class Exception: public std::exception {
    public:

        Exception(std::string message, std::string function, std::string file, size_t line, spbla_Info info, bool critical)
            : mMessage(std::move(message)),
              mFunction(std::move(function)),
              mFile(std::move(file)),
              mLine(line),
              mInfo(info),
              mCritical(critical) {

        }

        Exception(const Exception& e) noexcept = default;
        Exception(Exception&& e) noexcept = default;

        ~Exception() override = default;

        const char* what() const noexcept override {
            return mMessage.c_str();
        }

        const std::string& GetMessage() const {
            return mMessage;
        }

        const std::string& GetFunction() const {
            return mFunction;
        }

        const std::string& GetFile() const {
            return mFile;
        }

        size_t GetLine() const {
            return mLine;
        }

        spbla_Info GetInfo() const {
            return mInfo;
        }

        bool IsCritical() const {
            return mCritical;
        }

    private:
        std::string mMessage;
        std::string mFunction;
        std::string mFile;
        size_t mLine;
        spbla_Info mInfo;
        bool mCritical;
    };

    /**
     * Exceptions with spbla_Info error code parametrisation.
     * @tparam Type Exception error code (type)
     */
    template<spbla_Info Type>
    class TException: public Exception {
    public:
        TException(std::string&& message, std::string&& function, std::string&& file, size_t line, bool critical)
            : Exception(std::move(message), std::move(function), std::move(file), line, Type, critical)  {

        }

        TException(const TException& other) noexcept = default;
        TException(TException&& other) noexcept = default;

        ~TException() override = default;
    };

    // Errors exposed to the C API
    using Error = TException<spbla_Info::SPBLA_INFO_ERROR>;
    using DeviceError = TException<spbla_Info::SPBLA_INFO_DEVICE_ERROR>;
    using MemOpFailed = TException<spbla_Info::SPBLA_INFO_MEM_OP_FAILED>;
    using InvalidArgument = TException<spbla_Info::SPBLA_INFO_INVALID_ARGUMENT>;
    using InvalidState = TException<spbla_Info::SPBLA_INFO_INVALID_STATE>;
    using BackendNotSupported = TException<spbla_Info::SPBLA_INFO_BACKEND_NOT_SUPPORTED>;
    using NotImplemented = TException<spbla_Info::SPBLA_INFO_NOT_IMPLEMENTED>;

}

// An error, in theory, can recover after this
#define RAISE_ERROR(type, message)                                                      \
    do {                                                                                \
        throw ::spbla::type(message, __FUNCTION__, __FILE__, __LINE__, false);          \
    } while (0);

#define CHECK_RAISE_ERROR(condition, type, message)                                     \
    if (!(condition)) { RAISE_ERROR(type, #condition ": " message); } else { }

// Critical errors, cause library shutdown
#define RAISE_CRITICAL_ERROR(type, message)                                             \
    do {                                                                                \
        throw ::spbla::type(message, __FUNCTION__, __FILE__, __LINE__, true);           \
    } while (0);

#define CHECK_RAISE_CRITICAL_ERROR(condition, type, message)                            \
    if (!(condition)) { RAISE_CRITICAL_ERROR(type, #condition ": " message); } else { }

#endif //SPBLA_EXCEPTION_HPP