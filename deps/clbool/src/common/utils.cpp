#include "utils.hpp"

namespace clbool::utils {

    bool compare_buffers(Controls &controls, const cl::Buffer &buffer_gpu, const cpu_buffer &buffer_cpu, uint32_t size,
                         std::string name) {
        if (size != buffer_cpu.size()) {
            std::cerr << "size of buffers for " << name << " are different " << std::endl
                      << size << " on GPU vs " << buffer_cpu.size() << " on CPU " << std::endl;
            return false;
        }

        if (size == 0) return true;

        cpu_buffer cpu_copy(size);

        CLB_COPY_BUF(controls.queue.enqueueReadBuffer(buffer_gpu, CL_TRUE, 0, sizeof(uint32_t) * cpu_copy.size(),
                                                 cpu_copy.data()));

        for (uint32_t i = 0; i < size; ++i) {
            if (cpu_copy[i] != buffer_cpu[i]) {
                uint32_t start = std::max(0, (int) i - 10);
                uint32_t stop = std::min(size, i + 10);
                std::cerr << "buffers for " << name << " are different " << std::endl
                          << "{ i: (gpu[i], cpu[i]) }" << std::endl;
                for (uint32_t j = start; j < stop; ++j) {
                    if (j == i) {
                    std::cerr << " !!! " << j << ": (" << cpu_copy[j] << ", " << buffer_cpu[j] << "), ";
                    } else {
                        std::cerr << j << ": (" << cpu_copy[j] << ", " << buffer_cpu[j] << "), ";
                    }
                }
                std::cerr << std::endl;
                std::cerr << "buffers for " << name << " are different";
                return false;
            }
        }
        LOG << "buffers for " << name << " are equal";
        return true;
    }

    bool compare_matrices(Controls &controls, const matrix_dcsr &m_gpu, const matrix_dcsr_cpu &m_cpu) {
        if (m_gpu.nnz() != m_cpu.cols().size()) {
            std::cerr << "diff nnz, gpu: " << m_gpu.nnz() << " vs cpu: " << m_cpu.cols().size() << std::endl;
            return false;
        }
        if (m_gpu.nnz() == 0) {
            LOG << "Matrix is empty";
            return true;
        }

        return
                compare_buffers(controls, m_gpu.rpt_gpu(), m_cpu.rpt(), m_gpu.nzr() + 1, "rpt") &&
                compare_buffers(controls, m_gpu.rows_gpu(), m_cpu.rows(), m_gpu.nzr(), "rows") &&
                compare_buffers(controls, m_gpu.cols_gpu(), m_cpu.cols(), m_gpu.nnz(), "cols");
    }

    bool compare_matrices(Controls &controls, const matrix_csr &m_gpu, const matrix_csr_cpu &m_cpu) {
        if (m_gpu.nnz() != m_cpu.cols().size()) {
            std::cerr << "diff nnz, gpu: " << m_gpu.nnz() << " vs cpu: " << m_cpu.cols().size() << std::endl;
            return false;
        }
        if (m_gpu.nnz() == 0) {
            LOG << "Matrix is empty";
            return true;
        }

        return
                compare_buffers(controls, m_gpu.rpt_gpu(), m_cpu.rpt(), m_gpu.nrows() + 1, "rpt") &&
                compare_buffers(controls, m_gpu.cols_gpu(), m_cpu.cols(), m_gpu.nnz(), "cols");
    }


// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v) {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x) {
        x = x | (x >> 1);
        x = x | (x >> 2);
        x = x | (x >> 4);
        x = x | (x >> 8);
        x = x | (x >> 16);
        return x - (x >> 1);
    }

    std::string mapDeviceType(cl_device_type type) {
        switch (type) {
            case (1 << 1):
                return "CL_DEVICE_TYPE_CPU";
            case (1 << 2):
                return "CL_DEVICE_TYPE_GPU";
            case (1 << 3):
                return "CL_DEVICE_TYPE_ACCELERATOR";
            default:
                return "UNKNOWN";
        }
    }

    void printDeviceInfo(const cl::Device &device) {
        std::cout << "        CL_DEVICE_TYPE: " << mapDeviceType(device.getInfo<CL_DEVICE_TYPE>()) << std::endl;
        std::cout << "        CL_DEVICE_AVAILABLE: " << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
        std::cout << "        CL_DEVICE_LOCAL_MEM_SIZE: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
        std::cout << "        CL_DEVICE_GLOBAL_MEM_SIZE: "
                  << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << std::endl;
        std::cout << "        CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                  << std::endl;
        std::cout << "        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: "
                  << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;


    }

    void printPlatformInfo(const cl::Platform &platform) {
        std::vector<cl::Device> devices;
        std::cout << "CL_PLATFORM_PROFILE: " << platform.getInfo<CL_PLATFORM_PROFILE>() << std::endl;
        std::cout << "CL_PLATFORM_VERSION: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
        std::cout << "CL_PLATFORM_NAME: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "CL_PLATFORM_VENDOR: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (uint32_t i = 0; i < devices.size(); ++i) {
            std::cout << "        device id: " << i << "\n";
            printDeviceInfo(devices[i]);
        }

        std::cout << "-----------------------" << std::endl;
    }

    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size) {
        cpu_buffer cpu_copy(size);
        controls.queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(uint32_t) * cpu_copy.size(), cpu_copy.data());
        uint32_t counter = 0;
        for (auto const &item: cpu_copy) {
            std::cout << counter << ": " << item << ", ";
            counter++;
            if (counter % 200 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void print_cpu_buffer(const cpu_buffer &buffer, uint32_t size) {
        uint32_t end = size;
        if (size == -1) end = buffer.size();
        for (uint32_t i = 0; i < end; ++i) {
            std::cout << buffer[i] << ", ";
        }
        std::cout << std::endl;
    }


    void fill_random_buffer(cpu_buffer &buf, uint32_t max) {
        if (max <= 0 && max != -1) {
            throw std::runtime_error("Illegal argument, 13417565323");
        }
        uint32_t n = buf.size();
        FastRandom r(n);
        for (uint32_t i = 0; i < n; ++i) {
            buf[i] = max != -1 ? r.next(0, max) : r.next();
        }
    }

    cl::Buffer create_buffer(Controls &controls, uint32_t size) {
        return cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof (uint32_t) * size);
    }

    cl::Buffer create_buffer(Controls &controls, cpu_buffer &cpuBuffer, bool readonly) {
        return cl::Buffer(controls.queue, cpuBuffer.begin(), cpuBuffer.end(), readonly);
    }

    cl::Event read_buffer(Controls &controls, cpu_buffer &result, const cl::Buffer &source) {
        cl::Event ev;
        controls.queue.enqueueReadBuffer(source, false, 0, sizeof(uint32_t) * result.size(), result.data(),
                                         nullptr, &ev);
        return ev;
    }

}