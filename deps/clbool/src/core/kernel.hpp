#pragma once

#include "../common/utils.hpp"
#include "kernel_cache.hpp"

namespace clbool {

    /*
     * This class refers to a device kernel.
     * As *.cl file may consist of many kernels,
     * it needs file name and kernel name to identify certain kernel.
     *
     * @tparam Args Types of kernel arguments
     */
    template<typename ... Args>
    class kernel {
    public:
        using kernel_type = cl::KernelFunctor<Args...>;

    private:

        std::string _program_name;
        std::string _kernel_name;
        uint32_t _block_size = 0;
        uint32_t _req_work_size = 0;
        bool _async = false;

        std::string options_str;

        void check_completeness(const Controls &controls) {
            std::string suffix = "in kernel " + _kernel_name + " of program " + _program_name;
            if (_program_name == "") CLB_RAISE("no program name" + suffix, CLBOOL_INVALID_VALUE);
            if (_kernel_name == "") CLB_RAISE("no kernel name" + suffix, CLBOOL_INVALID_VALUE);
            if (_req_work_size == 0) CLB_RAISE("zero global_work_size" + suffix, CLBOOL_INVALID_VALUE);
            if (_block_size == 0) _block_size = controls.block_size;
        }

        uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n) {
            return (n + work_group_size - 1) / work_group_size * work_group_size;
        }

    public:

        kernel(std::string program_name, std::string kernel_name) noexcept
                : _program_name(program_name), _kernel_name(kernel_name) {}

        kernel &set_kernel_name(std::string kernel_name) {
            _kernel_name = std::move(kernel_name);
            return *this;
        }

        kernel &set_block_size(uint32_t block_size) {
            _block_size = block_size;
            return *this;
        }

        /*
         * @param req_work_size The minimal number of threads required for a kernel execution.
         * Doesn't necessarily multiple of block_size.
         */
        kernel &set_work_size(uint32_t req_work_size) {
            _req_work_size = req_work_size;
            return *this;
        }

        template<typename OptionType>
        kernel &add_option(std::string name, const OptionType &value) {
            options_str += (" -D " + name + "=" + std::to_string(value));
            return *this;
        }


        kernel &set_async(bool async) {
            _async = async;
            return *this;
        }


        cl::Event run(Controls &controls, Args ... args) {
            check_completeness(controls);
            // Option RUN allows not to include the clion_defines.cl file in kernel.
            std::string build_options = options_str + " -D RUN  -D GROUP_SIZE=" + std::to_string(_block_size);
            cl::Kernel kernel = details::KernelCache::get_kernel(controls, _program_name, _kernel_name,
                                                                 build_options);

            kernel_type functor(kernel);
            cl::EnqueueArgs eargs(_async ? controls.async_queue : controls.queue,
                                  cl::NDRange(calculate_global_size(_block_size, _req_work_size)),
                                  cl::NDRange(_block_size));

            return functor(eargs, args...);
        }
    };
}