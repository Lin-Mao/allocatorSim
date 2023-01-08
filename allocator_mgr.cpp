#include "allocator_mgr.h"
#include <cassert>
#include <iomanip>

namespace c10 {
namespace cuda {
namespace AllocatorSim {

allocatorMgr::allocatorMgr() : allocatorMgr(0, 0) {
    this->block_ref_map = std::map<Block*, size_t>();
    allocatorSim alloc_sim();
}

allocatorMgr::allocatorMgr(int device, int stream) {
    this->device = device;
    this->stream = stream;
    this->block_ref_map = std::map<Block*, size_t>();
    allocatorSim alloc_sim();
}

bool allocatorMgr::check_constraints() {
    if (allocatorConf::get_kMinLargeAlloc() >= allocatorConf::get_kLargeBuffer()) {
        return false;
    }

    return true;
}

template<typename FUNC1, typename FUNC2, typename candidate_t>
void allocatorMgr::search_candidates(FUNC1 get_func, FUNC2 set_func,
                                    std::set<candidate_t> candidates) {
    for (auto candidate : candidates) {
        auto prev = get_func();
        set_func(candidate);

        if (!check_constraints()) {
            set_func(prev);
            continue;
        }
        
        auto reserved_size = simulate_allocator().second;
        if (reserved_size >= current_reserved_size) {
            set_func(prev);
        }

        current_reserved_size = std::min(current_reserved_size, reserved_size);
        reset_allocator_memory_usage();
        empty_cache();
    }
}

void allocatorMgr::log_configs(Configs& configs) {
    auto memory_usage = simulate_allocator();
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

void allocatorMgr::collect_trace(void* ptr, int64_t size) {
    if (size > 0) {
        _active_blocks.emplace(ptr, std::make_pair(op_id, size));
        op_id++;
    } else {
        auto b = _active_blocks.find(ptr);
        _trace.emplace(
            b->second.first, std::make_pair(op_id, b->second.second));
        _active_blocks.erase(b);
        op_id++;
    }
}

char* allocatorMgr::malloc_cpu_memory_chunk(size_t size) {
    char* pointer = (char*) malloc(size);
    return pointer;
}

void allocatorMgr::free_cpu_memory_chunk(char* pointer) {
    free((void*)pointer);
}

bool allocatorMgr::iteration_trigger(bool begin, size_t active_size) {
    size_t result = false;
    if (begin) {
        // do something
    } else {
        if (initial_opt) {
            optimize_configs(5);
            result = true;
            initial_opt = false;
            std::cout << "===================================" << std::endl;
            std::cout << active_size << std::endl;
            std::cout << original_configs.reserved_size << std::endl;
            std::cout << searched_configs.reserved_size << std::endl;
            std::cout << "===================================" << std::endl;
            // _active_blocks.clear();
            apply_configs(original_configs);
        }
        _trace.clear();
    }
    return result;
}

std::pair<size_t, size_t> allocatorMgr::simulate_allocator() {
    for (uint64_t i = 0; i <= op_id; i++) {
        auto block = _trace.find(i);
        if (block != _trace.end()) {
            auto reference = std::get<0>(block->second) - block->first;
            malloc_block(std::get<1>(block->second), reference);
        }
        update_block_reference();
        free_block();
    }
    auto memory_usage = get_allocator_memory_usage();
    assert(memory_usage.first <= memory_usage.second);
    return memory_usage;
}

void allocatorMgr::optimize_configs(int nums) {
    log_configs(original_configs);

    for (int i = 0; i < nums; i++) {
        search_candidates(
            allocatorConf::get_kMinBlockSize,
            allocatorConf::set_kMinBlockSize,
            kMinBlockSize_candidates);
        
        search_candidates(
            allocatorConf::get_kSmallSize,
            allocatorConf::set_kSmallSize,
            kSmallSize_candidates);

        search_candidates(
            allocatorConf::get_kSmallBuffer,
            allocatorConf::set_kSmallBuffer,
            kSmallBuffer_candidates);

        search_candidates(
            allocatorConf::get_kLargeBuffer,
            allocatorConf::set_kLargeBuffer,
            kLargeBuffer_candidates);

        search_candidates(
            allocatorConf::get_kMinLargeAlloc,
            allocatorConf::set_kMinLargeAlloc,
            kMinLargeAlloc_candidates);

        search_candidates(
            allocatorConf::get_kRoundLarge,
            allocatorConf::set_kRoundLarge,
            kRoundLarge_candidates);
    }

    log_configs(searched_configs);
    // report_configs();
}

void allocatorMgr::report_configs() {
    int width = 36;
    std::cout << std::setw(width) << std::left << "###################### [Config result] ######################"
              << std::endl;
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
    std::cout << "############################################################" << std::endl;
}

void allocatorMgr::apply_configs(const Configs& configs) {
    allocatorConf::set_kMinBlockSize(configs.kMinBlockSize);
    allocatorConf::set_kSmallSize(configs.kSmallSize);
    allocatorConf::set_kSmallBuffer(configs.kSmallBuffer);
    allocatorConf::set_kLargeBuffer(configs.kLargeBuffer);
    allocatorConf::set_kMinLargeAlloc(configs.kMinLargeAlloc);
    allocatorConf::set_kRoundLarge(configs.kRoundLarge);
    allocatorConf::set_max_split_size(configs.m_max_split_size);
    allocatorConf::set_roundup_power2_divisions(configs.m_roundup_power2_divisions);
    allocatorConf::set_roundup_bypass_threshold(configs.m_roundup_bypass_threshold);
    allocatorConf::set_garbage_collection_threshold(configs.m_garbage_collection_threshold);
    allocatorConf::set_memory_segment_address_start(configs.m_memory_segment_address_start);
    allocatorConf::set_memory_segment_address_interval(configs.m_memory_segment_address_interval);
}

void allocatorMgr::malloc_block(size_t orig_size, size_t ref) {
    Block* b = this->alloc_sim.malloc(this->device, orig_size, this->stream);
    this->block_ref_map.emplace(b, ref);
}

void allocatorMgr::update_block_reference() {
    for (auto& b : this->block_ref_map) {
        --b.second;
    }
}

void allocatorMgr::free_block() {
    std::vector<Block*> del_blocks;
    for (const auto b : this->block_ref_map) {
        if (b.second == 0) {
            this->alloc_sim.free(b.first);
            del_blocks.push_back(b.first);
        }
    }

    for (const auto b : del_blocks) {
        this->block_ref_map.erase(b);
    }
}

void allocatorMgr::empty_cache() {
    alloc_sim.release_cached_blocks();
}

std::pair<size_t, size_t> allocatorMgr::get_allocator_memory_usage() {
    return alloc_sim.get_max_memory_usage();
}

void allocatorMgr::reset_allocator_memory_usage() {
    alloc_sim.set_max_memory_usage(0, 0);
}

void allocatorMgr::show_allocator_memory_usage() {
    auto memory_usage = get_allocator_memory_usage();
    std::cout << "Max allocated size: " << memory_usage.first << " B ("
              << format_size(memory_usage.first) << ")" << std::endl;
    std::cout << "Max reserved size: " << memory_usage.second << " B ("
              << format_size(memory_usage.second) << ")" << std::endl;
}

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim