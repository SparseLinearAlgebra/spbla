#include "env.hpp"

#include <sstream>


namespace clbool {
    Controls create_controls(uint32_t platform_id, uint32_t device_id) {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        std::vector<cl::Kernel> kernels;
        cl::Program program;
        cl::Device device;
        try {
            cl::Platform::get(&platforms);
            if (platform_id >= platforms.size()) {
                std::stringstream s;
                s << "No such platform: " << platform_id
                << ". Run show_devices() to enumerate available platforms and devices.";
                CLB_RAISE(s.str(), CLBOOL_INITIALIZATION_ERROR);
            }
            platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (device_id >= devices.size()) {
                std::stringstream s;
                s << "No such device: " << platform_id
                << ". Run show_devices() to enumerate available platforms and devices.";
                CLB_RAISE(s.str(), CLBOOL_INITIALIZATION_ERROR);
            }
            uint32_t max_wg_size = devices[device_id].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            return Controls(devices[device_id], max_wg_size);

        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << e.err() << "\n";
            throw std::runtime_error(exception.str());
        }
    }

    void show_devices() {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Kernel> kernels;
        cl::Program program;
        cl::Device device;
        try {
            cl::Platform::get(&platforms);
            for (size_t i = 0; i < platforms.size(); ++i) {
                std::cout << "platform id: " << i << " \n";
                utils::printPlatformInfo(platforms[i]);
            }
        } catch (const cl::Error &e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << e.err() << "\n";
            throw std::runtime_error(exception.str());
        }
    }
}

