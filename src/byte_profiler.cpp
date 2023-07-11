#include "byte_profiler.h"
#include "allocator_utils.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <atomic>


namespace c10 {
namespace cuda {
namespace ByteProfiler {
namespace {
    int device_index = 0;
    int max_step_monitored = 10;
    std::atomic<int> total_finished(0);
    int num_devices = 8;

    std::unordered_map<cudaStream_t, int> stream2int;
}  // anonymous namespace for variables

void set_max_step(int max_step) {
    max_step_monitored = max_step;
}

device_allocator::device_allocator() {
    std::cout << "Device allocator of device " << device_index << " is created." << std::endl;
    this->device = device_index;
    device_index++;

    max_step = max_step_monitored;

    auto pca_dir = std::getenv("PYTORCH_PCA_TRACE_DIR");
    if (pca_dir) {
        path = pca_dir;
        std::cout << "PYTORCH_PCA_TRACE_DIR is set to " << path << std::endl;
    } else {
        path = "./output/";
        std::cout << "PYTORCH_PCA_TRACE_DIR is not set. Use default value: ./output/" << std::endl;
    }

    // in case the directory is not created
    if (!fs::is_directory(path)) {
        int ret = system(("mkdir -p " + path).c_str());
        if (ret != 0) {}
    }
    memory_file = "/device" + std::to_string(this->device) + "_memory" + ".csv";
    std::ofstream output(path + memory_file);
    output << "global_id,stream_id,size,allocated_cur,reserved_cur" << std::endl;
    output.close();
    
}

device_allocator::~device_allocator() {
    std::ofstream output(path + memory_file, std::ios::app);
    output << std::endl;
    output << "max_allocated_size," << max_allocated_size << std::endl;
    output << "max_reserved_size," << max_reserved_size << std::endl;
    output.close();
}

void device_allocator::collect_memory_usage(cudaStream_t stream, int64_t size,
                                            size_t allocated_cur, size_t reserved_cur) {
    if (step_id >= max_step) {
        return;
    }

    int stream_id;
    if (stream2int.find(stream) == stream2int.end()) {
        stream_id  = stream2int.size();
        stream2int.emplace(stream, stream_id);
    } else {
        stream_id = stream2int[stream];
    }

    std::ofstream output(path + memory_file, std::ios::app);
    output << global_id << "," << stream_id << "," << size << ","
           << allocated_cur << "," << reserved_cur << std::endl;
    output.close();
    global_id++;
    
    max_allocated_size = std::max(max_allocated_size, allocated_cur);
    max_reserved_size = std::max(max_reserved_size, reserved_cur);
}

void device_allocator::step_end() {
    if (step_id >= max_step) {
        total_finished++;
        if (total_finished == num_devices) {
            std::cout << "All devices have finished their monitoring." << std::endl;
            exit(0);
        }
    }
    std::cout << "Monitor step " << step_id << " ends." << std::endl;
    std::ofstream output(path + memory_file, std::ios::app);
    output << "<<<<<<<<<< step " << step_id << " end >>>>>>>>>>" << std::endl;
    step_id++;
    output.close();
}


}  // namespace ByteProfiler
}  // namespace cuda
}  // namespace c10

