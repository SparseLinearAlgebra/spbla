#pragma once

#include "CL/opencl.hpp"

#include <exception>
#include <string>
#include <sstream>
#include <unordered_map>


namespace clbool {

    inline std::string error_name(cl_int error) {
        switch (error) {
            case 0:
                return "CL_SUCCESS";
            case -1:
                return "CL_DEVICE_NOT_FOUND";
            case -2:
                return "CL_DEVICE_NOT_AVAILABLE";
            case -3:
                return "CL_COMPILER_NOT_AVAILABLE";
            case -4:
                return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case -5:
                return "CL_OUT_OF_RESOURCES";
            case -6:
                return "CL_OUT_OF_HOST_MEMORY";
            case -7:
                return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case -8:
                return "CL_MEM_COPY_OVERLAP";
            case -9:
                return "CL_IMAGE_FORMAT_MISMATCH";
            case -10:
                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case -11:
                return "CL_BUILD_PROGRAM_FAILURE";
            case -12:
                return "CL_MAP_FAILURE";
            case -13:
                return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case -14:
                return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case -15:
                return "CL_COMPILE_PROGRAM_FAILURE";
            case -16:
                return "CL_LINKER_NOT_AVAILABLE";
            case -17:
                return "CL_LINK_PROGRAM_FAILURE";
            case -18:
                return "CL_DEVICE_PARTITION_FAILED";
            case -19:
                return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            case -30:
                return "CL_INVALID_VALUE";
            case -31:
                return "CL_INVALID_DEVICE_TYPE";
            case -32:
                return "CL_INVALID_PLATFORM";
            case -33:
                return "CL_INVALID_DEVICE";
            case -34:
                return "CL_INVALID_CONTEXT";
            case -35:
                return "CL_INVALID_QUEUE_PROPERTIES";
            case -36:
                return "CL_INVALID_COMMAND_QUEUE";
            case -37:
                return "CL_INVALID_HOST_PTR";
            case -38:
                return "CL_INVALID_MEM_OBJECT";
            case -39:
                return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case -40:
                return "CL_INVALID_IMAGE_SIZE";
            case -41:
                return "CL_INVALID_SAMPLER";
            case -42:
                return "CL_INVALID_BINARY";
            case -43:
                return "CL_INVALID_BUILD_OPTIONS";
            case -44:
                return "CL_INVALID_PROGRAM";
            case -45:
                return "CL_INVALID_PROGRAM_EXECUTABLE";
            case -46:
                return "CL_INVALID_KERNEL_NAME";
            case -47:
                return "CL_INVALID_KERNEL_DEFINITION";
            case -48:
                return "CL_INVALID_KERNEL";
            case -49:
                return "CL_INVALID_ARG_INDEX";
            case -50:
                return "CL_INVALID_ARG_VALUE";
            case -51:
                return "CL_INVALID_ARG_SIZE";
            case -52:
                return "CL_INVALID_KERNEL_ARGS";
            case -53:
                return "CL_INVALID_WORK_DIMENSION";
            case -54:
                return "CL_INVALID_WORK_GROUP_SIZE";
            case -55:
                return "CL_INVALID_WORK_ITEM_SIZE";
            case -56:
                return "CL_INVALID_GLOBAL_OFFSET";
            case -57:
                return "CL_INVALID_EVENT_WAIT_LIST";
            case -58:
                return "CL_INVALID_EVENT";
            case -59:
                return "CL_INVALID_OPERATION";
            case -60:
                return "CL_INVALID_GL_OBJECT";
            case -61:
                return "CL_INVALID_BUFFER_SIZE";
            case -62:
                return "CL_INVALID_MIP_LEVEL";
            case -63:
                return "CL_INVALID_GLOBAL_WORK_SIZE";
            case -64:
                return "CL_INVALID_PROPERTY";
            default:
                return "unknown error code: " + std::to_string(error);
        }
    }

    enum Status {
        CLBOOL_NO_SUCH_PROGRAM,
        CLBOOL_CREATE_KERNEL_ERROR,
        CLBOOL_BUILD_PROGRAM_ERROR,

        CLBOOL_INCOMPLETE_KERNEL,
        CLBOOL_RUN_KERNEL_FAILURE,

        CLBOOL_EVENT_WAITING_ERROR,

        CLBOOL_CREATE_BUFFER_ERROR,
        CLBOOL_COPY_BUFFER_ERROR,
        CLBOOL_READ_BUFFER_ERROR,
        CLBOOL_WRITE_BUFFER_ERROR,

        CLBOOL_CREATE_CONTROLS_ERROR,

        CLBOOL_INVALID_ARGUMENT,
        CLBOOL_INVALID_VALUE,

        CLBOOL_INITIALIZATION_ERROR
    };

    // https://stackoverflow.com/a/3342891
    inline std::ostream &operator<<(std::ostream &out, const Status s) {
        static std::unordered_map<Status, std::string> strings;
        if (strings.empty()) {
#define INSERT_ELEMENT(p) strings[p] = #p
            INSERT_ELEMENT(CLBOOL_NO_SUCH_PROGRAM);
            INSERT_ELEMENT(CLBOOL_CREATE_KERNEL_ERROR);
            INSERT_ELEMENT(CLBOOL_BUILD_PROGRAM_ERROR);

            INSERT_ELEMENT(CLBOOL_INCOMPLETE_KERNEL);
            INSERT_ELEMENT(CLBOOL_RUN_KERNEL_FAILURE);

            INSERT_ELEMENT(CLBOOL_EVENT_WAITING_ERROR);

            INSERT_ELEMENT(CLBOOL_CREATE_BUFFER_ERROR);
            INSERT_ELEMENT(CLBOOL_COPY_BUFFER_ERROR);
            INSERT_ELEMENT(CLBOOL_READ_BUFFER_ERROR);
            INSERT_ELEMENT(CLBOOL_WRITE_BUFFER_ERROR);

            INSERT_ELEMENT(CLBOOL_CREATE_CONTROLS_ERROR);

            INSERT_ELEMENT(CLBOOL_INVALID_ARGUMENT);
            INSERT_ELEMENT(CLBOOL_INVALID_VALUE);

            INSERT_ELEMENT(CLBOOL_INITIALIZATION_ERROR);
#undef INSERT_ELEMENT
        }
        return out << strings[s];
    }

    class Exception : public std::exception {
    private:

        mutable std::string message;
        Status status;
        std::string file;
        std::string function;
        size_t line;

    public:
        Exception(std::string message, Status status, std::string file,
                  std::string function, size_t line)
                : message(std::move(message)), status(status), file(std::move(file)),
                  function(std::move(function)), line(line) {

        }

        const char *what() const noexcept override {
            std::stringstream s;

            s << "[" << status << "] " << "\"" << message << "\""<< std::endl
              << file << ": line: " << line << " function: " << function << std::endl;

            message = s.str();
            return message.c_str();
        }

        std::string get_message() const {
            return message;
        }
    };

//
    inline std::string clError_handler(const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << error_name(e.err()) << " \n";
        return exception.str();
    }

    inline std::string
    program_handler(const cl::Error &e, const cl::Program &program, const std::string &pname,  const cl::Device &device) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << error_name(e.err()) << " \n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << "Error while build " << pname << std::endl;
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        return exception.str();
    }


#define CLB_CL(expr, status) try {                                                        \
    expr;                                                                                                   \
    }                                                                                                       \
    catch (const cl::Error &e) {                                                                            \
         throw clbool::Exception(clError_handler(e), status,                                        \
         __FILE__, __FUNCTION__, __LINE__);                                                                 \
    }

#define CLB_CHECK(cond, msg, status) do {                                               \
            if (!(cond)) throw  Exception(msg, status, __FILE__, __FUNCTION__, __LINE__); \
} while (0);

#define CLB_CREATE_BUF(expr) CLB_CL(expr, CLBOOL_CREATE_BUFFER_ERROR)
#define CLB_COPY_BUF(expr) CLB_CL(expr, CLBOOL_COPY_BUFFER_ERROR)

#define CLB_BUILD(build) try {                                                                    \
    build;                                                                                                          \
    }                                                                                                               \
    catch (const cl::Error &e)  {                                                                                   \
        throw Exception(program_handler(e, cl_program, program_name, controls.device), CLBOOL_BUILD_PROGRAM_ERROR,\
        __FILE__, __FUNCTION__, __LINE__);                                                                          \
    }

#define CLB_RUN(run) try {                                                                           \
        run;                                                                                         \
    } catch (const Exception& e) {                                                                   \
      throw Exception(e.get_message(), CLBOOL_RUN_KERNEL_FAILURE, __FILE__, __FUNCTION__, __LINE__); \
} catch (const cl::Error &e) {                                                                       \
         throw clbool::Exception(clError_handler(e), CLBOOL_RUN_KERNEL_FAILURE,                                         \
         __FILE__, __FUNCTION__, __LINE__);                                                          \
                                                                                                     \
}
#define CLB_WAIT(wait) CLB_CL(wait, CLBOOL_EVENT_WAITING_ERROR)
#define CLB_WRITE_BUF(write) CLB_CL(write, CLBOOL_WRITE_BUFFER_ERROR)
#define CLB_READ_BUF(read) CLB_CL(read, CLBOOL_READ_BUFFER_ERROR)
#define CLB_RAISE(msg, status) do { \
      throw Exception(msg, status, __FILE__, __FUNCTION__, __LINE__);                                           \
} while(0);
}