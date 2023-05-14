#include "allocator_utils.h"
#include <iostream>

namespace c10 {
namespace cuda {
namespace AllocatorSim {


namespace {
// the op_id is used in the whole code, include simulator and allocator
op_id_t global_op_id = 0;

std::string dump_file_path = "/home/lm/torch/torch1/pytorch/third_party/allocatorSim/output/";

}  // anonymous namespace

op_id_t get_global_op_id() {
    return global_op_id;
}
void increase_global_op_id() {
    global_op_id++;
}

std::string get_dump_file_path() {
    return dump_file_path;
}

std::array<size_t, TIMER_NUMS> allocatorTimer::timers
                                    = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
std::array<std::string, TIMER_NUMS> allocatorTimer::timer_names
                                    = {"", "", "", "", "", "", "", "", "", ""};
std::array<sys_clock, TIMER_NUMS> allocatorTimer::starts = {};
std::array<sys_clock, TIMER_NUMS> allocatorTimer::ends = {};

void allocatorTimer::start_timer(int index) {
    starts[index] = std::chrono::system_clock::now();
}

void allocatorTimer::stop_timer(int index) {
    ends[index] = std::chrono::system_clock::now();
}

void allocatorTimer::log_timer(int index, std::string name) {
    if (timer_names[index] == "") {
        timer_names[index] = name;
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(ends[index] - starts[index]);
    timers[index] += double(duration.count()) * std::chrono::microseconds::period::num;
}

void allocatorTimer::print_timer(int index) {
    std::cout << std::string(timer_names[index]) << ": " << timers[index] << " us" << std::endl;
}

std::string format_size(size_t size) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024) {
        os << size << " bytes";
    } else if (size <= 1048576) {
        os << (size / 1024.0);
        os << " KB";
    } else if (size <= 1073741824ULL) {
        os << size / 1048576.0;
        os << " MB";
    } else {
        os << size / 1073741824.0;
        os << " GB";
    }
    return os.str();
}

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim
