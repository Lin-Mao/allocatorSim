#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <tuple>

#include "allocatorSim.h"

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

int main() {
    allocatorSim allocSim;
    allocSim.test_allocator();

    std::vector<std::tuple<size_t, size_t, size_t>> input_block_list;

    std::ifstream file;
    file.open("./input/large_block_test.log");
    std::string line;
    while (getline(file, line)) {
        auto vec = split_line(line, " ");
        input_block_list.push_back(std::make_tuple(vec[0], vec[1], vec[2]));
    }

    std::for_each(input_block_list.begin(), input_block_list.end(),
    [] (std::tuple<size_t, size_t, size_t> i) {
        std::cout << std::get<0>(i) << ", " << std::get<1>(i) << ", "
                  << std::get<2>(i) << std::endl;
    });

    return 0;
}
