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
        
        auto reserved_size = simulate_allocator();
        if (reserved_size >= current_reserved_size) {
            set_func(prev);
        } else {
            log_configs(searched_configs, false);
        }

        current_reserved_size = std::min(current_reserved_size, reserved_size);
        reset_allocator_memory_usage();
        empty_cache();
    }
}

bool allocatorMgr::check_configs(Configs& config) {
    if (config.kMinLargeAlloc >= config.kLargeBuffer) {
        return false;
    }

    return true;
}

void allocatorMgr::search_group() {
    log_configs(original_configs);
    reset_allocator_memory_usage();
    empty_cache();
    for (auto diff : GROUP_DIFFERENCES) {
        group_blocks(diff);
        if (evaluate_allocator(original_configs, original_configs)) {
            current_difference = diff;
            allocatorConf::BACKUP_GROUPS = allocatorConf::_GROUPS;
            alloc_sim.set_group_enable_flag_sim(true);
            group_enable_flag = true;
        } else if(group_enable_flag) {
            // rollback
            allocatorConf::_GROUPS = allocatorConf::BACKUP_GROUPS;
        }
        reset_allocator_memory_usage();
        empty_cache();
    }
    evaluate_allocator(original_configs, original_configs);
    log_configs(searched_configs);
    report_configs(original_configs, searched_configs);
    if (group_enable_flag) {
        std::cout << "Group difference: " << current_difference << std::endl;
    } else {
        std::cout << "No group" << std::endl;
    }
}

void allocatorMgr::allocator_assert(bool expr) {
    if (expr) {
        return;
    } else {
        report_configs(original_configs, searched_configs);
        assert(expr);
    }
}

bool allocatorMgr::evaluate_allocator(Configs configs, Configs prev_conf) {
    apply_configs(configs);
    auto reserved_size = simulate_allocator();
    if (reserved_size < current_reserved_size) {
        current_reserved_size = std::min(current_reserved_size, reserved_size);
        std::cout << "reserved size: " << reserved_size << std::endl;
        return true;
    } else {
        apply_configs(prev_conf);
    }
    return false;
}

void allocatorMgr::search_config() {
    log_configs(original_configs);
    reset_allocator_memory_usage();
    empty_cache();
    auto prev_conf = original_configs;
    for (auto kMinBlockSize : kMinBlockSize_candidates) {
        for (auto kSmallSize : kSmallSize_candidates) {
            for (auto kSmallBuffer : kSmallBuffer_candidates) {
                for (auto kLargeBuffer : kLargeBuffer_candidates) {
                    for (auto kMinLargeAlloc : kMinLargeAlloc_candidates) {
                        for (auto kRoundLarge : kRoundLarge_candidates) {
                            searched_configs = Configs(
                                kMinBlockSize,
                                kSmallSize,
                                kSmallBuffer,
                                kLargeBuffer,
                                kMinLargeAlloc,
                                kRoundLarge, 0, 0);
                            if(!check_configs(searched_configs)) {
                                continue;
                            }

                            if (evaluate_allocator(searched_configs, prev_conf)) {
                                prev_conf = searched_configs;
                            }
                            reset_allocator_memory_usage();
                            empty_cache();
                        }

                    }

                }
            }
        }
    }
    apply_configs(prev_conf);
    evaluate_allocator(prev_conf, prev_conf);
    log_configs(searched_configs);
    report_configs(original_configs, searched_configs);

    // search_config_with_group();
}

// after search group
void allocatorMgr::search_config_with_group() {
    log_configs(original_configs);
    reset_allocator_memory_usage();
    empty_cache();
    auto prev_conf = searched_configs;
    for (auto kMinBlockSize : kMinBlockSize_candidates) {
        for (auto kSmallSize : kSmallSize_candidates) {
            for (auto kSmallBuffer : kSmallBuffer_candidates) {
                for (auto kLargeBuffer : kLargeBuffer_candidates) {
                    for (auto kMinLargeAlloc : kMinLargeAlloc_candidates) {
                        for (auto kRoundLarge : kRoundLarge_candidates) {
                            searched_configs = Configs(
                                kMinBlockSize,
                                kSmallSize,
                                kSmallBuffer,
                                kLargeBuffer,
                                kMinLargeAlloc,
                                kRoundLarge, 0, 0);
                            if(!check_configs(searched_configs)) {
                                continue;
                            }
                            for (auto diff : GROUP_DIFFERENCES) {
                                group_blocks(diff);
                                if (evaluate_allocator(searched_configs, prev_conf)) {
                                    prev_conf = searched_configs;
                                    current_difference = diff;
                                    allocatorConf::BACKUP_GROUPS = allocatorConf::_GROUPS;
                                    alloc_sim.set_group_enable_flag_sim(true);
                                    group_enable_flag = true;
                                } else if(group_enable_flag) {
                                    // rollback
                                    allocatorConf::_GROUPS = allocatorConf::BACKUP_GROUPS;
                                }
                                reset_allocator_memory_usage();
                                empty_cache();
                            }
                        }

                    }

                }
            }
        }
    }
    apply_configs(prev_conf);
    evaluate_allocator(prev_conf, prev_conf);
    log_configs(searched_configs);
    report_configs(original_configs, searched_configs);
}

void allocatorMgr::search_configs() {
    log_configs(original_configs);
    for (size_t i = 0; i < CONFIG_NUMS; i++) {
        auto candidates = ALL_CANDIDATES[i];
        auto set_func = allocatorConf::set_funcs[i];
        auto get_func = allocatorConf::get_funcs[i];
        for (auto candidate : candidates) {
            auto prev = get_func();
            set_func(candidate);

            if (!check_constraints()) {
                set_func(prev);
                continue;
            }

            for(size_t j = 0; j < CONFIG_NUMS; j++) {
                if (j == i) continue;
                search_candidates(
                    allocatorConf::get_funcs[j],
                    allocatorConf::set_funcs[j],
                    ALL_CANDIDATES[j]);
            }
        }
    }
    apply_configs(searched_configs);
    log_configs(searched_configs);
    report_configs(original_configs, searched_configs);
}

void allocatorMgr::log_configs(Configs& configs, bool get_mem) {
    size_t reserved_size = 0;
    size_t allocated_size = 0;
    if (get_mem) {
        reserved_size = get_reserved_bytes();
        allocated_size = get_allocated_bytes();
    }
    configs = Configs(
        allocatorConf::get_kMinBlockSize(),
        allocatorConf::get_kSmallSize(),
        allocatorConf::get_kSmallBuffer(),
        allocatorConf::get_kLargeBuffer(),
        allocatorConf::get_kMinLargeAlloc(),
        allocatorConf::get_kRoundLarge(),
        reserved_size,
        allocated_size
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
            current_reserved_size = simulate_allocator();
            // search_config();
            // search_group();
            search_config_with_group();
            result = true;
            initial_opt = false;
            _active_blocks.clear();
        }
        _trace.clear();
    }
    return result;
}

size_t allocatorMgr::simulate_allocator() {
    for (uint64_t i = 0; i <= op_id; i++) {
        auto block = _trace.find(i);
        if (block != _trace.end()) {
            auto reference = std::get<0>(block->second) - block->first;
            malloc_block(std::get<1>(block->second), reference);
        }
        update_block_reference();
        free_block();
    }

    auto reserved_size = get_reserved_bytes();
    auto allocated_size = get_allocated_bytes();

    allocator_assert(reserved_size > allocated_size);
    return reserved_size;
}

void allocatorMgr::report_configs(const Configs& conf_before, const Configs& conf_after) {
    int width = 36;
    std::cout << std::setw(width) << std::left
              << "###################### [Config result] ######################" << std::endl;
    std::cout << std::setw(width) << std::left << "Max allocated size: "
              << static_cast<int64_t>(conf_before.allocated_size) << " => "
              << static_cast<int64_t>(conf_after.allocated_size) << " diff: "
              << static_cast<int64_t>(conf_before.allocated_size - conf_after.allocated_size)
              << std::endl;
    std::cout << std::setw(width) << std::left << "Max reserved size: "
              << static_cast<int64_t>(conf_before.reserved_size) << " => "
              << static_cast<int64_t>(conf_after.reserved_size) << " diff: "
              << static_cast<int64_t>(conf_before.reserved_size - conf_after.reserved_size)
              << std::endl;
    std::cout << std::setw(width) << std::left << "kMinBlockSize: " << conf_before.kMinBlockSize << " => "
              << conf_after.kMinBlockSize << std::endl;
    std::cout << std::setw(width) << std::left << "kSmallSize: " << conf_before.kSmallSize << " => "
              << conf_after.kSmallSize << std::endl;
    std::cout << std::setw(width) << std::left << "kSmallBuffer: " << conf_before.kSmallBuffer << " => "
              << conf_after.kSmallBuffer << std::endl;
    std::cout << std::setw(width) << std::left << "kLargeBuffer: " << conf_before.kLargeBuffer << " => "
              << conf_after.kLargeBuffer << std::endl;
    std::cout << std::setw(width) << std::left << "kMinLargeAlloc: " << conf_before.kMinLargeAlloc << " => "
              << conf_after.kMinLargeAlloc << std::endl;
    std::cout << std::setw(width) << std::left << "kRoundLarge: " << conf_before.kRoundLarge << " => "
              << conf_after.kRoundLarge << std::endl;
    std::cout << std::setw(width) << std::left << "m_max_split_size: " << conf_before.m_max_split_size
              << " => " << conf_after.m_max_split_size << std::endl;
    std::cout << std::setw(width) << std::left << "m_roundup_power2_divisions: "
              << conf_before.m_roundup_power2_divisions << " => "
              << conf_after.m_roundup_power2_divisions << std::endl;
    std::cout << std::setw(width) << std::left << "m_roundup_bypass_threshold: "
              << conf_before.m_roundup_bypass_threshold << " => "
              << conf_after.m_roundup_bypass_threshold << std::endl;
    std::cout << std::setw(width) << std::left << "m_garbage_collection_threshold: "
              << conf_before.m_garbage_collection_threshold << " => "
              << conf_after.m_garbage_collection_threshold << std::endl;
    std::cout << std::setw(width) << std::left << "m_memory_segment_address_start: "
              << conf_before.m_memory_segment_address_start << " => "
              << conf_after.m_memory_segment_address_start << std::endl;
    std::cout << std::setw(width) << std::left << "m_memory_segment_address_interval: "
              << conf_before.m_memory_segment_address_interval << " => "
              << conf_after.m_memory_segment_address_interval << std::endl;
    std::cout << "############################################################" << std::endl;
}

void allocatorMgr::apply_configs(const Configs& configs) {
    allocatorConf::set_kMinBlockSize(configs.kMinBlockSize);
    allocatorConf::set_kSmallSize(configs.kSmallSize);
    allocatorConf::set_kSmallBuffer(configs.kSmallBuffer);
    allocatorConf::set_kLargeBuffer(configs.kLargeBuffer);
    allocatorConf::set_kMinLargeAlloc(configs.kMinLargeAlloc);
    allocatorConf::set_kRoundLarge(configs.kRoundLarge);
    // allocatorConf::set_max_split_size(configs.m_max_split_size);
    // allocatorConf::set_roundup_power2_divisions(configs.m_roundup_power2_divisions);
    // allocatorConf::set_roundup_bypass_threshold(configs.m_roundup_bypass_threshold);
    // allocatorConf::set_garbage_collection_threshold(configs.m_garbage_collection_threshold);
    // allocatorConf::set_memory_segment_address_start(configs.m_memory_segment_address_start);
    // allocatorConf::set_memory_segment_address_interval(configs.m_memory_segment_address_interval);
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

size_t allocatorMgr::get_reserved_bytes() {
    return alloc_sim.get_max_reserved_bytes();
}

size_t allocatorMgr::get_allocated_bytes() {
    return alloc_sim.get_max_allocated_bytes();
}

void allocatorMgr::reset_allocator_memory_usage() {
    alloc_sim.set_max_memory_usage(0, 0);
}

void allocatorMgr::show_allocator_memory_usage() {
    std::pair<size_t, size_t> memory_usage = get_allocator_memory_usage();
    std::cout << "Max allocated size: " << memory_usage.first << " B ("
              << format_size(memory_usage.first) << ")" << std::endl;
    std::cout << "Max reserved size: " << memory_usage.second << " B ("
              << format_size(memory_usage.second) << ")" << std::endl;
}

void allocatorMgr::group_blocks(const float& difference) {
    std::set<size_t> block_sizes;
    for (auto t : _trace) {
        if (t.second.second > allocatorConf::get_kLargeBuffer()) {
            block_sizes.insert(t.second.second);
        }
    }

    for (int i = 0; i < GROUP_NUMS; i++) {
        allocatorConf::_GROUPS[i] = std::numeric_limits<size_t>::max();
    }
    size_t small_group_size = *block_sizes.begin();
    size_t group_boundary = 0;
    int index = 0;
    for (auto it = block_sizes.begin(); it != block_sizes.end();) {
        if ((*it - small_group_size) / small_group_size > difference) {
            group_boundary = *std::prev(it);
            allocatorConf::_GROUPS[index] =group_boundary;
            index++;
            small_group_size = *it;
            if (index == GROUP_NUMS-1) {
                allocatorConf::_GROUPS[index] = *block_sizes.rbegin();
                index++;
                break;
            }
        }
        it++;
    }
    if (!block_sizes.empty() && group_boundary != *block_sizes.rbegin()) {
        allocatorConf::_GROUPS[index] = *block_sizes.rbegin();
    }
}

size_t allocatorMgr::get_grouped_allocation_size(size_t size) {
    if (size < allocatorConf::_GROUPS[0]) {
        if (allocatorConf::_GROUPS[0] != std::numeric_limits<size_t>::max()) {
            return allocatorConf::_GROUPS[0];
        } else {
            auto tunablekRoundLarge = AllocatorSim::allocatorConf::get_kRoundLarge();
            return tunablekRoundLarge * ((size + tunablekRoundLarge - 1) / tunablekRoundLarge);
        }
    } else if (size < allocatorConf::_GROUPS[1]) {
        if (allocatorConf::_GROUPS[1] != std::numeric_limits<size_t>::max()) {
            return allocatorConf::_GROUPS[1];
        } else {
            auto tunablekRoundLarge = AllocatorSim::allocatorConf::get_kRoundLarge();
            return tunablekRoundLarge * ((size + tunablekRoundLarge - 1) / tunablekRoundLarge);
        }
    } else if (size < allocatorConf::_GROUPS[2]) {
        if (allocatorConf::_GROUPS[2] != std::numeric_limits<size_t>::max()) {
            return allocatorConf::_GROUPS[2];
        } else {
            auto tunablekRoundLarge = AllocatorSim::allocatorConf::get_kRoundLarge();
            return tunablekRoundLarge * ((size + tunablekRoundLarge - 1) / tunablekRoundLarge);
        }
    } else if (size < allocatorConf::_GROUPS[3]) {
        if (allocatorConf::_GROUPS[3] != std::numeric_limits<size_t>::max()) {
            return allocatorConf::_GROUPS[3];
        } else {
            auto tunablekRoundLarge = AllocatorSim::allocatorConf::get_kRoundLarge();
            return tunablekRoundLarge * ((size + tunablekRoundLarge - 1) / tunablekRoundLarge);
        }
    } else if (size < allocatorConf::_GROUPS[4]) {
        if (allocatorConf::_GROUPS[4] != std::numeric_limits<size_t>::max()) {
            return allocatorConf::_GROUPS[4];
        } else {
            auto tunablekRoundLarge = AllocatorSim::allocatorConf::get_kRoundLarge();
            return tunablekRoundLarge * ((size + tunablekRoundLarge - 1) / tunablekRoundLarge);
        }
    } else {
        auto tunablekRoundLarge = AllocatorSim::allocatorConf::get_kRoundLarge();
        return tunablekRoundLarge * ((size + tunablekRoundLarge - 1) / tunablekRoundLarge);
    }
}

size_t allocatorMgr::get_allocation_size(size_t size) {
    if (group_enable_flag && size > allocatorConf::get_kLargeBuffer()) {
        return get_grouped_allocation_size(size);
    }
    auto tunablekSmallSize = allocatorConf::get_kSmallSize();
    if (size <= tunablekSmallSize) {
        return allocatorConf::get_kSmallBuffer();
    } else if (size < allocatorConf::get_kMinLargeAlloc()) {
        return allocatorConf::get_kLargeBuffer();
    } else {
        auto tunablekRoundLarge = allocatorConf::get_kRoundLarge();
        return tunablekRoundLarge * ((size + tunablekRoundLarge - 1) / tunablekRoundLarge);
    }
}


}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim