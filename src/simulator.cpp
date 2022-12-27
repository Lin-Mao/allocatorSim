#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <tuple>

#include "allocator_sim.h"
#include "allocator_mgr.h"

typedef std::vector<std::tuple<uint64_t, uint64_t, size_t>> blockVector_t;

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

std::tuple<uint64_t, uint64_t> process_trace(std::string filename, blockVector_t& block_list) {
    std::ifstream file;
    file.open(filename);
    std::string line;
    while (getline(file, line)) {
        auto vec = split_line(line, " ");
        block_list.push_back(std::make_tuple(vec[0], vec[1], vec[2]));
    }
    file.close();

    uint64_t min = std::numeric_limits<uint64_t>::max();
    uint64_t max = 0;
    std::for_each(block_list.begin(), block_list.end(),
    [&min, &max](std::tuple<uint64_t, uint64_t, size_t> t) {
        if (std::get<0>(t) < min) {
            min = std::get<0>(t);
        }
        if (std::get<1>(t) > max) {
            max = std::get<1>(t);
        }
    });

    /*
    std::for_each(block_list.begin(), block_list.end(),
    [] (std::tuple<size_t, size_t, size_t> i) {
        std::cout << std::get<0>(i) << ", " << std::get<1>(i) << ", "
                  << std::get<2>(i) << std::endl;
    });
    */
    return std::make_tuple(min, max);
}

void run_allocator(const blockVector_t& block_list, const uint64_t min, const uint64_t max) {
    allocatorMgr alloc_mgr;

    uint64_t count = 0;
    for (uint64_t i = min; i <= max; i++) {
        if (std::get<0>(block_list[count]) == i) {
            auto ref = std::get<1>(block_list[count]) - std::get<0>(block_list[count]);
            alloc_mgr.malloc_block(std::get<2>(block_list[count]), ref);
            count++;
        }
        alloc_mgr.update_block_reference();
        alloc_mgr.free_block();
    }
}

int main() {
    // allocatorSim allocSim;
    // allocSim.test_allocator();

    blockVector_t input_block_list;
    uint64_t min, max;
    std::tie(min, max) = process_trace("./input/small_block_test.log", input_block_list);
    run_allocator(input_block_list, min, max);

    return 0;
}
