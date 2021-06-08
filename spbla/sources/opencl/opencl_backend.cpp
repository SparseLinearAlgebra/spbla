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

#include <opencl/opencl_backend.hpp>
#include <core/error.hpp>
#include <controls.hpp>
#include <utils.hpp>
#include <env.hpp>
#include <opencl/opencl_matrix.hpp>
#include <regex>

namespace spbla {

    namespace oclDetails {
        template<int size>
        int getNameMaxLength(char(&)[size]){return size;}
    }

    std::shared_ptr<clbool::Controls> OpenCLBackend::controls = nullptr;

    void OpenCLBackend::initialize(hints initHints) {
        if (!isInitialized())
            controls = std::make_shared<clbool::Controls>(clbool::create_controls(0, 0));
    }

    void OpenCLBackend::finalize() {
    }

    bool OpenCLBackend::isInitialized() const {
        return controls != nullptr;
    }

    MatrixBase* OpenCLBackend::createMatrix(size_t nrows, size_t ncols) {
        return new OpenCLMatrix(controls.get(), nrows, ncols);
    }

    void OpenCLBackend::releaseMatrix(MatrixBase *matrixBase) {
        delete matrixBase;
    }

    std::pair<int, int> OpenCLBackend::getVersion() {
        int major = -1;
        int minor = -1;
        // OpenCL 1.2 CUDA
        auto versonStr = controls->device.getInfo<CL_DEVICE_VERSION>();

        std::string::size_type pos = versonStr.find(' ');
        if (pos == std::string::npos) return {major, minor};
        // 1.2 CUDA
        versonStr = versonStr.substr(pos + 1);

        pos = versonStr.find(' ');
        if (pos == std::string::npos) return {major, minor};
        // 1.2
        versonStr = versonStr.substr(0, pos);

        pos = versonStr.find('.');
        if (pos == std::string::npos) return {major, minor};

        try {
            major = std::stoi(versonStr.substr(0, pos));
        } catch (...) {
            return {major, minor};
        }

        try {
            minor = std::stoi(versonStr.substr(pos + 1));
        } catch (...) {
            return {major, minor};
        }

        return {major, minor};
    }

    int OpenCLBackend::getWarp() {
        static std::regex nvidiaRegex("NVIDIA", std::regex_constants::icase);
        static std::regex amdRegex("AMD", std::regex_constants::icase);
        std::string vendor = controls->device.getInfo<CL_DEVICE_VENDOR>();
        if (std::regex_search(vendor, nvidiaRegex)) return OpenCLBackend::NVIDIA_WARP;
        if (std::regex_search(vendor, amdRegex)) return OpenCLBackend::AMD_WARP;
        return -1;
    }

    void OpenCLBackend::queryCapabilities(spbla_DeviceCaps &caps) {
        if (controls != nullptr) {

            {
                int maxNameLength = oclDetails::getNameMaxLength(caps.name);
                std::string nameStr = controls->device.getInfo<CL_DEVICE_NAME>();
                for (int i = 0; i < std::min(maxNameLength, (int) nameStr.size()); ++i) {
                    caps.name[i] = nameStr[i];
                }
                for (int i = std::min(maxNameLength, (int) nameStr.size()); i < maxNameLength; ++i) {
                    caps.name[i] = '\0';
                }
            }

            caps.cudaSupported = false;
            caps.openclSupported = true;


            auto version = getVersion();
            caps.major = version.first;
            caps.minor = version.second;

            caps.warp = getWarp();

            caps.globalMemoryKiBs = controls->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024;
            caps.sharedMemoryPerBlockKiBs = controls->device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024;
            caps.sharedMemoryPerMultiProcKiBs = caps.sharedMemoryPerBlockKiBs;

        }
    }

    void OpenCLBackend::queryAvailableDevices() {
        clbool::show_devices();
    }

}