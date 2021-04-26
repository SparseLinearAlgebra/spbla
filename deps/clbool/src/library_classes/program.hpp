#pragma once

#include "../common/utils.hpp"

namespace clbool {

    template <typename ... Args>
    class program {
    public:
        using kernel_type = cl::KernelFunctor<Args...>;

    private:
        const char * _kernel = "";
        uint32_t _kernel_length = 0;
        std::string _kernel_name;
        uint32_t _block_size = 0;
        uint32_t _needed_work_size = 0;
        cl::Program cl_program;
        bool _built = false;
        bool _async = false;

        std::string options_str;

        void check_completeness() {
            if (_kernel_length == 0) throw std::runtime_error("zero kernel length");
            if (_kernel_name == "") throw std::runtime_error("no kernel name");
            if (_needed_work_size == 0) throw std::runtime_error("zero global_work_size");
        }

        void build_cl_program(Controls &controls) {
            if (_block_size == 0) _block_size = controls.block_size;
            try {
                cl_program = controls.create_program_from_source(_kernel, _kernel_length);
                std::stringstream options;
                options <<  options_str << " -D RUN " << " -D GROUP_SIZE=" << _block_size;
                cl_program.build(options.str().c_str());

            } catch (const cl::Error &e) {
                utils::program_handler(e, cl_program, controls.device, _kernel_name);
            }
        }

    public:
        program() = default;
        explicit program(const char *kernel, uint32_t kernel_length)
        : _kernel(kernel)
        , _kernel_length(kernel_length)
        {}

        program& set_sources(const char *kernel, uint32_t kernel_length) {
            _kernel = kernel;
            _kernel_length = kernel_length;
            _built = false;
            return *this;
        }

        program& set_kernel_name(std::string kernel_name) {
            _kernel_name = std::move(kernel_name);
            return *this;
        }

        program& set_block_size(uint32_t block_size) {
            _block_size = block_size;
            _built = false;
            return *this;
        }

        program& set_needed_work_size(uint32_t needed_work_size) {
            _needed_work_size = needed_work_size;
            return *this;
        }

        program& add_option(std::string name, std::string value = "") {
            options_str += (" -D " + name + "=" + value);
            _built = false;
            return *this;
        }

        template<typename OptionType>
        program& add_option(std::string name, const OptionType &value) {
            options_str += (" -D " + name + "=" + std::to_string(value));
            _built = false;
            return *this;
        }

        program& set_async(bool async) {
            _async = async;
            return *this;
        }

        void build(Controls &controls) {
            build_cl_program(controls);
            _built = true;
        }


        cl::Event run(Controls &controls, Args ... args) {
            check_completeness();
            try {
                if (!_built) {
                    build_cl_program(controls);
                    _built = true;
                }
                cl::Kernel kernel(cl_program, _kernel_name.c_str());

                kernel_type functor(kernel);

                cl::EnqueueArgs eargs(_async ? controls.async_queue : controls.queue,
                                      cl::NDRange(utils::calculate_global_size(_block_size, _needed_work_size)),
                                      cl::NDRange(_block_size));

                return functor(eargs, args...);
            } catch (const cl::Error &e) {
                utils::program_handler(e, cl_program, controls.device, _kernel_name);
            }
        }
    };
}