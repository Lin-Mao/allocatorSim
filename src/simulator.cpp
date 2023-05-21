#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <tuple>

#include "allocator_manager.h"

using trace_type_t = c10::cuda::AllocatorSim::trace_t;
using trace_type = std::map<uint64_t, std::pair<uint64_t, size_t>>;

std::vector<size_t> split_line(std::string str, const std::string c) {
    std::vector<size_t> vec;

    std::string::size_type pos1, pos2;
    pos1 = 0;
    pos2 = str.find(c);
    while (pos2 != std::string::npos) {
        vec.push_back(std::stol(str.substr(pos1, pos2 - pos1)));

        pos1 = pos2 + c.size();
        pos2 = str.find(c, pos1);
    }

    if (pos1 != str.length()) {
        vec.push_back(std::stol(str.substr(pos1)));
    }

    return vec;
}

std::pair<uint64_t, uint64_t> process_trace(std::string filename, trace_type_t& block_map) {
    std::ifstream file;
    file.open(filename);
    std::string line;
    while (getline(file, line)) {
        auto vec = split_line(line, " ");
        block_map.emplace(vec[0], std::make_pair(vec[1], vec[2]));
    }
    file.close();

    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    std::for_each(block_map.begin(), block_map.end(),
    [&min, &max](std::pair<uint64_t, std::pair<uint64_t, size_t>> p) {
        if (std::get<0>(p) < min) {
            min = std::get<0>(p);
        }
        if (std::get<0>(std::get<1>(p)) > max) {
            max = std::get<0>(std::get<1>(p));
        }
    });

    return std::make_pair(min, max);
}

void generate_trace(const trace_type_t& block_map, trace_type& malloc_map, trace_type& free_map) {
    uint64_t ptr = c10::cuda::AllocatorSim::allocatorConf::get_memory_segment_address_start();
    uint64_t interval = c10::cuda::AllocatorSim::allocatorConf::get_memory_segment_address_interval();

    std::for_each(block_map.begin(), block_map.end(),
    [&malloc_map, &free_map, &ptr, &interval](std::pair<uint64_t, std::pair<uint64_t, size_t>> p) {
        malloc_map.emplace(p.first, std::make_pair(ptr, p.second.second));
        free_map.emplace(p.second.first, std::make_pair(ptr, p.second.second));
        ptr += interval;
    });
}

void run_allocator(const trace_type& malloc_map, const trace_type& free_map, uint64_t min, uint64_t max) {
    c10::cuda::AllocatorSim::allocatorMgr alloc_mgr;

    for (uint64_t i = min; i <= max; i++) {
        auto alloc = malloc_map.find(i);
        if (alloc != malloc_map.end()) {
            alloc_mgr.collect_trace(
                reinterpret_cast<void*>(alloc->second.first), static_cast<int64_t>(alloc->second.second));
        }
        auto free = free_map.find(i);
        if (free != free_map.end()) {
            alloc_mgr.collect_trace(
                reinterpret_cast<void*>(free->second.first), static_cast<int64_t>(-free->second.second));
        }
    }

    alloc_mgr.test_simulator();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./bin/allocatorsim <trace_file> <allocator_config_file>" << std::endl;
        return 0;
    }

    // // trace format(each line): start_op_id end_op_id tensor_size
    // std::string trace_file = "./input/alexnet_train.log";
    // std::string config_file = "./input/allocator_config.json";

    std::string trace_file = argv[1];
    std::string config_file = argv[2];

    trace_type_t input_block_map;
    trace_type malloc_map;
    trace_type free_map;
    uint64_t min, max;

    std::tie(min, max) = process_trace(trace_file, input_block_map);

    generate_trace(input_block_map, malloc_map, free_map);
    run_allocator(malloc_map, free_map, min, max);

    return 0;
}
