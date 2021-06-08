#pragma once

#include "../cl/headers/headers_map.hpp"
#include "error.hpp"

namespace clbool::details {
    using program_id = std::string; // kernel name|options
    using kernel_id = std::pair<program_id, std::string>; // program_id,  kernel name

    struct KernelCache {
        struct pair_hash {
            template<class T1, class T2>
            std::size_t operator()(const std::pair<T1, T2> &p) const {
                auto h1 = std::hash<T1>{}(p.first);
                auto h2 = std::hash<T2>{}(p.second);
                return h1 ^ h2;
            }
        };

        static inline std::unordered_map<program_id, cl::Program> programs{};
        static inline std::unordered_map<kernel_id, cl::Kernel, pair_hash> kernels{};

        static const cl::Program &get_program(const Controls &controls, const std::string &program_name,
                                              const std::string &options = "") {
            cl::Program cl_program;
            std::string program_key = program_name + "|" + options;
            if (KernelCache::programs.find(program_key) != KernelCache::programs.end()) {
                return programs[program_key];
            }

            auto source_ptr = HeadersMap.find(program_name);
            CLB_CHECK(source_ptr != HeadersMap.end(), "Cannot find " + program_name, CLBOOL_NO_SUCH_PROGRAM);

            {
                START_TIMING
                KernelSource source = source_ptr->second;
                cl_program = cl::Program(controls.context, std::vector<std::basic_string<char>>{{source.kernel, source.length}});
                CLB_BUILD(cl_program.build(options.c_str()));
                KernelCache::programs[program_key] = cl_program;
                END_TIMING(" kernel " + program_name + " build in: ")
            }

            return KernelCache::programs[program_key];
        }

        static const cl::Kernel &get_kernel(const Controls &controls, const std::string &program_name,
                                            const std::string &kernel_name, const std::string &options) {
            cl::Program cl_program = get_program(controls, program_name, options);
            kernel_id kernelId = {program_name + "|" + options, kernel_name};
            if (kernels.find(kernelId) != kernels.end()) {
                return kernels[kernelId];
            }
            CLB_CL(kernels[kernelId] = cl::Kernel(cl_program, kernel_name.c_str()),
                     CLBOOL_CREATE_KERNEL_ERROR);
            return kernels[kernelId];
        }

    };
}