#include "allocator_opt.h"

allocatorOpt::allocatorOpt(blockMap_t trace, uint64_t max, uint64_t min)
                        : trace(trace), max(max), min(min) {
    current_max_size = evaluate_model();
}

size_t allocatorOpt::evaluate_model() {
    for (uint64_t i = min; i <= max; i++) {
        auto block = trace.find(i);
        if (block != trace.end()) {
            auto reference = std::get<0>(block->second) - block->first;
            alloc_mgr.malloc_block(std::get<1>(block->second), reference);
        }
        alloc_mgr.update_block_reference();
        alloc_mgr.free_block();
    }

    return alloc_mgr.get_reserved_size();
}

void allocatorOpt::search_kMinBlockSize() {
    for (auto candidate : kMinBlockSize_candidates) {
        auto prev = allocatorConf::get_kMinBlockSize();
        allocatorConf::set_kMinBlockSize(candidate);

        auto reserved_size = evaluate_model();
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kMinBlockSize(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_reserved_size();
        alloc_mgr.empty_cache();
    }
}
    
void allocatorOpt::search_kSmallSize() {
    for (auto candidate : kSmallSize_candidates) {
        auto prev = allocatorConf::get_kSmallSize();
        allocatorConf::set_kSmallSize(candidate);

        auto reserved_size = evaluate_model();
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kSmallSize(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_reserved_size();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kSmallBuffer() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kSmallBuffer();
        allocatorConf::set_kSmallBuffer(candidate);

        auto reserved_size = evaluate_model();
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kSmallBuffer(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_reserved_size();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kLargeBuffer() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kLargeBuffer();

        allocatorConf::set_kLargeBuffer(candidate);
        auto reserved_size = evaluate_model();
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kLargeBuffer(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_reserved_size();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kMinLargeAlloc() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kMinLargeAlloc();
        allocatorConf::set_kMinLargeAlloc(candidate);

        auto reserved_size = evaluate_model();
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kMinLargeAlloc(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_reserved_size();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_kRoundLarge() {
    for (auto candidate : kSmallBuffer_candidates) {
        auto prev = allocatorConf::get_kRoundLarge();
        allocatorConf::set_kRoundLarge(candidate);

        auto reserved_size = evaluate_model();
        if (reserved_size >= current_max_size) {
            allocatorConf::set_kRoundLarge(prev);
        }
        current_max_size = std::min(current_max_size, reserved_size);
        alloc_mgr.reset_reserved_size();
        alloc_mgr.empty_cache();
    }
}

void allocatorOpt::search_configs() {
    search_kMinBlockSize();
    search_kSmallSize();
    search_kSmallBuffer();
    search_kLargeBuffer();
    search_kMinLargeAlloc();
    search_kRoundLarge();
    evaluate_model();

    report_config();
}

void allocatorOpt::report_config() {
    std::cout << std::endl << "[Config result]" << std::endl;
    std::cout << "kMinBlockSize: " << allocatorConf::get_kMinBlockSize() << std::endl;
    std::cout << "kSmallSize: " << allocatorConf::get_kSmallSize() << std::endl;
    std::cout << "kSmallBuffer: " << allocatorConf::get_kSmallBuffer() << std::endl;
    std::cout << "kLargeBuffer: " << allocatorConf::get_kLargeBuffer() << std::endl;
    std::cout << "kMinLargeAlloc: " << allocatorConf::get_kMinLargeAlloc() << std::endl;
    std::cout << "kRoundLarge: " << allocatorConf::get_kRoundLarge() << std::endl;
    std::cout << "m_max_split_size: " << allocatorConf::get_max_split_size() << std::endl;
    std::cout << "m_roundup_power2_divisions: " << allocatorConf::get_roundup_power2_divisions() << std::endl;
    std::cout << "m_roundup_bypass_threshold: " << allocatorConf::get_roundup_bypass_threshold() << std::endl;
    std::cout << "m_garbage_collection_threshold: " << allocatorConf::get_garbage_collection_threshold() << std::endl;
    std::cout << "m_memory_segment_address_start: " << allocatorConf::get_memory_segment_address_start() << std::endl;
    std::cout << "m_memory_segment_address_interval: " << allocatorConf::get_memory_segment_address_interval() << std::endl;
}
