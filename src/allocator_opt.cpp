#include "allocator_opt.h"
#include <iomanip>

allocatorOpt::allocatorOpt(blockMap_t trace, uint64_t max, uint64_t min)
                        : trace(trace), max(max), min(min) {
    current_max_size = evaluate_model().second;
}

std::pair<size_t, size_t> allocatorOpt::evaluate_model() {
    for (uint64_t i = min; i <= max; i++) {
        auto block = trace.find(i);
        if (block != trace.end()) {
            auto reference = std::get<0>(block->second) - block->first;
            alloc_mgr.malloc_block(std::get<1>(block->second), reference);
        }
        alloc_mgr.update_block_reference();
        alloc_mgr.free_block();
    }

    return alloc_mgr.get_allocator_memory_usage();
}

void allocatorOpt::search_kMinBlockSize() {
    for (auto candidate : kMinBlockSize_candidates) {
        auto prev = allocatorConf::get_kMinBlockSize();
        allocatorConf::set_kMinBlockSize(candidate);

        auto reserved_size = evaluate_model().second;
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kMinBlockSize(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_allocator_memory_usage();
        alloc_mgr.empty_cache();
    }
}
    
void allocatorOpt::search_kSmallSize() {
    for (auto candidate : kSmallSize_candidates) {
        auto prev = allocatorConf::get_kSmallSize();
        allocatorConf::set_kSmallSize(candidate);

        auto reserved_size = evaluate_model().second;
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kSmallSize(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_allocator_memory_usage();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kSmallBuffer() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kSmallBuffer();
        allocatorConf::set_kSmallBuffer(candidate);

        auto reserved_size = evaluate_model().second;
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kSmallBuffer(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_allocator_memory_usage();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kLargeBuffer() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kLargeBuffer();

        allocatorConf::set_kLargeBuffer(candidate);
        auto reserved_size = evaluate_model().second;
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kLargeBuffer(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_allocator_memory_usage();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kMinLargeAlloc() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kMinLargeAlloc();
        allocatorConf::set_kMinLargeAlloc(candidate);

        auto reserved_size = evaluate_model().second;
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kMinLargeAlloc(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_allocator_memory_usage();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kRoundLarge() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kRoundLarge();
        allocatorConf::set_kRoundLarge(candidate);

        auto reserved_size = evaluate_model().second;
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kRoundLarge(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_allocator_memory_usage();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::log_configs(Configs& configs) {
    auto memory_usage = evaluate_model();
    configs = Configs(
        allocatorConf::get_kMinBlockSize(),
        allocatorConf::get_kSmallSize(),
        allocatorConf::get_kSmallBuffer(),
        allocatorConf::get_kLargeBuffer(),
        allocatorConf::get_kMinLargeAlloc(),
        allocatorConf::get_kRoundLarge(),
        memory_usage.first,
        memory_usage.second
    );
}

void allocatorOpt::search_configs() {
    log_configs(original_configs);
    
    search_kMinBlockSize();
    search_kSmallSize();
    search_kSmallBuffer();
    search_kLargeBuffer();
    search_kMinLargeAlloc();
    search_kRoundLarge();

    log_configs(searched_configs);
    report_config();
}

void allocatorOpt::report_config() {
    int width = 36;
    std::cout << std::setw(width) << std::left << "[Config result]" << std::endl;
    std::cout << std::setw(width) << std::left << "Max allocated size: " << original_configs.allocated_size
              << " => " << searched_configs.allocated_size << " diff: "
              << static_cast<int64_t>(original_configs.allocated_size - searched_configs.allocated_size)
              << std::endl;
    std::cout << std::setw(width) << std::left << "Max reserved size: " << original_configs.reserved_size
              << " => " << searched_configs.reserved_size << " diff: "
              << static_cast<int64_t>(original_configs.reserved_size - searched_configs.reserved_size)
              << std::endl;
    std::cout << std::setw(width) << std::left << "kMinBlockSize: " << original_configs.kMinBlockSize << " => "
              << searched_configs.kMinBlockSize << std::endl;
    std::cout << std::setw(width) << std::left << "kSmallSize: " << original_configs.kSmallSize << " => "
              << searched_configs.kSmallSize << std::endl;
    std::cout << std::setw(width) << std::left << "kSmallBuffer: " << original_configs.kSmallBuffer << " => "
              << searched_configs.kSmallBuffer << std::endl;
    std::cout << std::setw(width) << std::left << "kLargeBuffer: " << original_configs.kLargeBuffer << " => "
              << searched_configs.kLargeBuffer << std::endl;
    std::cout << std::setw(width) << std::left << "kMinLargeAlloc: " << original_configs.kMinLargeAlloc << " => "
              << searched_configs.kMinLargeAlloc << std::endl;
    std::cout << std::setw(width) << std::left << "kRoundLarge: " << original_configs.kRoundLarge << " => "
              << searched_configs.kRoundLarge << std::endl;
    std::cout << std::setw(width) << std::left << "m_max_split_size: " << original_configs.m_max_split_size
              << " => " << searched_configs.m_max_split_size << std::endl;
    std::cout << std::setw(width) << std::left << "m_roundup_power2_divisions: "
              << original_configs.m_roundup_power2_divisions << " => "
              << searched_configs.m_roundup_power2_divisions << std::endl;
    std::cout << std::setw(width) << std::left << "m_roundup_bypass_threshold: "
              << original_configs.m_roundup_bypass_threshold << " => "
              << searched_configs.m_roundup_bypass_threshold << std::endl;
    std::cout << std::setw(width) << std::left << "m_garbage_collection_threshold: "
              << original_configs.m_garbage_collection_threshold << " => "
              << searched_configs.m_garbage_collection_threshold << std::endl;
    std::cout << std::setw(width) << std::left << "m_memory_segment_address_start: "
              << original_configs.m_memory_segment_address_start << " => "
              << searched_configs.m_memory_segment_address_start << std::endl;
    std::cout << std::setw(width) << std::left << "m_memory_segment_address_interval: "
              << original_configs.m_memory_segment_address_interval << " => "
              << searched_configs.m_memory_segment_address_interval << std::endl;
}
