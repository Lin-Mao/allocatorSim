/**
 * This code (byte_profiler.h and byte_profiler.cpp) is dedicated to the byte branch,
 * which is used to pinpoint the memory issues (the unexpected gap between allocated and reserved memory)
 * during the LLM training. This code is a independent module and does not interact with original simulator.
 * It supports multiple-card scenarios in ByteDance with faster profiling speed.
**/
#ifndef BYTE_PROFILER_H
#define BYTE_PROFILER_H

#include <cuda_runtime.h>
#include <string>

namespace c10 {
namespace cuda {
namespace ByteProfiler {

typedef uint64_t op_id_t;

class device_allocator{
    int device;
    op_id_t global_id = 0;
    int step_id = 0;
    int max_step;

    size_t max_reserved_size = 0;
    size_t max_allocated_size = 0;

    std::string memory_file;

public:
    device_allocator();
    ~device_allocator();

    void collect_memory_usage(cudaStream_t stream, int64_t size, size_t allocated_cur, size_t reserved_cur);

    void step_end();
}; // class device_allocator

void set_max_step(int max_step);


}  // namespace ByteProfiler
}  // namespace cuda
}  // namespace c10

#endif // BYTE_PROFILER_H