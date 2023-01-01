#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <tuple>

#include "allocator_sim.h"
#include "allocator_mgr.h"
#include "allocator_opt.h"

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

std::pair<uint64_t, uint64_t> process_trace(std::string filename, blockMap_t& block_map) {
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

void run_allocator(const blockMap_t& block_map, const uint64_t min, const uint64_t max) {
    allocatorMgr alloc_mgr;

    for (uint64_t i = min; i <= max; i++) {
        auto block = block_map.find(i);
        if (block != block_map.end()) {
            auto reference = std::get<0>(block->second) - block->first;
            alloc_mgr.malloc_block(std::get<1>(block->second), reference);
        }
        alloc_mgr.update_block_reference();
        alloc_mgr.free_block();
    }
    alloc_mgr.show_allocator_memory_usage();
}

void search_config(const blockMap_t& block_map, const uint64_t min, const uint64_t max) {
    allocatorOpt alloc_opt(block_map, max, min);

    alloc_opt.search_configs();
}

int main() {
    std::string trace_file = "/home/lm/allocatorSim/input/baseline/sim_input.log";
    blockMap_t input_block_map;
    uint64_t min, max;
    std::tie(min, max) = process_trace(trace_file, input_block_map);
    // run_allocator(input_block_map, min, max);
    search_config(input_block_map, min, max);

    return 0;
}
