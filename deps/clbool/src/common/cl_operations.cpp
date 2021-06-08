#include "cl_operations.hpp"

namespace clbool {
    void prefix_sum(Controls &controls,
                    cl::Buffer &array,
                    uint32_t &total_sum,
                    uint32_t array_size) {
        auto scan = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("prefix_sum", "scan_blelloch");

        auto update = kernel<cl::Buffer, cl::Buffer, uint32_t, uint32_t>
                ("prefix_sum", "update_pref_sum");

        // число потоков, которое нужно выпустить для обработки массива ядром scan в данном алгоритме
        static auto threads_for_array = [](uint32_t size) -> uint32_t { return (size + 1) / 2; };
        // на каждом цикле wile число элементов, которые нужно обработать, сократится в times раз
        static auto reduce_array_size = [](uint32_t size, uint32_t times) -> uint32_t {
            return (size + times - 1) / times;
        };


        uint32_t block_size = controls.block_size; //controls.block_size;
        uint32_t d_block_size = 2 * block_size;
        scan.set_block_size(block_size);
        uint32_t a_size = reduce_array_size(array_size, d_block_size); // to save first roots
        uint32_t b_size = reduce_array_size(a_size, d_block_size); // to save second roots


        cl::Buffer a_gpu;
        CLB_CREATE_BUF(
        a_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_size));
        cl::Buffer b_gpu;
        CLB_CREATE_BUF(
        b_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * b_size));
        cl::Buffer total_sum_gpu;
        CLB_CREATE_BUF(
        total_sum_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t)));
        // массив будет уменьшаться в 2 * block_size раз.
        uint32_t outer = reduce_array_size(array_size, d_block_size);
        cl::Buffer *a_gpu_ptr = &a_gpu;
        cl::Buffer *b_gpu_ptr = &b_gpu;

        unsigned int *a_size_ptr = &a_size;
        unsigned int *b_size_ptr = &b_size;


        uint32_t leaf_size = 1;

        scan.set_work_size(threads_for_array(array_size));


        {
            START_TIMING
            CLB_RUN(scan.run(controls, a_gpu, array, total_sum_gpu, array_size).wait());
            END_TIMING("first scan: ")
        }


        while (outer > 1) {
            // subarray with pref sum
            leaf_size *= d_block_size;

            scan.set_work_size(threads_for_array(outer));

            {
                START_TIMING
                scan.run(controls, *b_gpu_ptr, *a_gpu_ptr, total_sum_gpu, outer).wait();
                END_TIMING("scan: ")
            };


            update.set_work_size(array_size - leaf_size);

            {
                START_TIMING
                update.run(controls, array, *a_gpu_ptr, array_size, leaf_size).wait();
                END_TIMING("update: ")
            }


            outer = reduce_array_size(outer, d_block_size);
            std::swap(a_gpu_ptr, b_gpu_ptr);
            std::swap(a_size_ptr, b_size_ptr);
        }

        CLB_READ_BUF(controls.queue.enqueueReadBuffer(total_sum_gpu, CL_TRUE, 0, sizeof(uint32_t), &total_sum));

    }

    void fill_with_zeroes(Controls &controls, cl::Buffer &array, uint32_t size) {
        fill_with(controls, array, size, 0);
    }

    void fill_with(Controls &controls, cl::Buffer &array, uint32_t size, uint32_t value) {
        auto fill = kernel<cl::Buffer, uint32_t, uint32_t>
                ("initialization", "fill_with");
        fill.set_block_size(controls.max_wg_size);
        fill.set_work_size(size);
        CLB_RUN(fill.run(controls, array, size, value).wait());
    }

}