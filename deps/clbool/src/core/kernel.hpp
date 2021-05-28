#pragma once

#include "../common/utils.hpp"
#include "kernel_cache.hpp"

namespace clbool {

    template <typename ... Args>
    class kernel {
    public:
        using kernel_type = cl::KernelFunctor<Args...>;

    private:
        const char * _kernel = "";
        uint32_t _kernel_length = 0;
        std::string _program_name;
        std::string _kernel_name;
        uint32_t _block_size = 0;
        uint32_t _needed_work_size = 0;
        bool _async = false;

        std::string options_str;

        void check_completeness(const Controls& controls) {
            if (_program_name == "") throw std::runtime_error("no kernel name");
            if (_kernel_name == "") throw std::runtime_error("no kernel name");
            if (_needed_work_size == 0) throw std::runtime_error("zero global_work_size");
            if (_block_size == 0) _block_size = controls.block_size;
        }

    public:
        kernel() = default;

        explicit kernel(std::string program_name,
                        std::string kernel_name)
        : _program_name(program_name)
        , _kernel_name(kernel_name)
        {}

        kernel& set_kernel_name(std::string kernel_name) {
            _kernel_name = std::move(kernel_name);
            return *this;
        }

        kernel& set_block_size(uint32_t block_size) {
            _block_size = block_size;
            return *this;
        }

        kernel& set_needed_work_size(uint32_t needed_work_size) {
            _needed_work_size = needed_work_size;
            return *this;
        }

        kernel& add_option(std::string name, std::string value = "") {
            options_str += (" -D " + name + "=" + value);
            return *this;
        }

        template<typename OptionType>
        kernel& add_option(std::string name, const OptionType &value) {
            options_str += (" -D " + name + "=" + std::to_string(value));
            return *this;
        }

        kernel& set_async(bool async) {
            _async = async;
            return *this;
        }


        cl::Event run(Controls &controls, Args ... args) {
            SET_TIMER
            check_completeness(controls);
            std::string build_options = options_str + " -D RUN  -D GROUP_SIZE=" + std::to_string(_block_size);
            try {
                cl::Kernel kernel = details::KernelCache::get_kernel(controls,
                            _program_name, _kernel_name, build_options);

                kernel_type functor(kernel);
                cl::EnqueueArgs eargs(_async ? controls.async_queue : controls.queue,
                                      cl::NDRange(utils::calculate_global_size(_block_size, _needed_work_size)),
                                      cl::NDRange(_block_size));

                return functor(eargs, args...);
            } catch (const cl::Error &e) {
                utils::program_handler(e, details::KernelCache::get_program(controls, build_options), controls.device, _kernel_name);
            }
        }
    };
}