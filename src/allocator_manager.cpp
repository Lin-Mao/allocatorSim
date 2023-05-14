#include "allocator_manager.h"
#include <cassert>
#include <iomanip>
#include "utils/hash.h"
#include "utils/python_states.h"
#include "utils/unwind_utils.h"
#include "utils/sanitizer_api.h"

#include <fstream>

namespace c10 {
namespace cuda {
namespace AllocatorSim {

namespace {
    bool enable_sync_collect_trace = true;

    bool is_profiling_mode = false;

    bool group_enable_flag = false;

    size_t iteration = 0;
    // <callpath_hash, <total_size, num_allocations>>
    std::map<std::string, std::pair<size_t, size_t>> static_tensor_callpaths;
    std::set<std::string> reclaimed_callpaths;
    std::map<void*, std::string> ptr2callpath;
    std::string dump_file_name = "optimized_configs.txt";


    std::set<std::string> unique_hash_trace;

    Configs original_configs;
    Configs searched_configs;
}   // namespace

const static size_t MAX_NUM_STATES = 30;
thread_local static python_state_t python_states[MAX_NUM_STATES];

void load_opt_guidance(std::string filename) {
    std::ifstream in(filename);
    assert(in.good() && "Cannot open the file");
    in >> searched_configs.kMinBlockSize;
    in >> searched_configs.kSmallSize;
    in >> searched_configs.kSmallBuffer;
    in >> searched_configs.kLargeBuffer;
    in >> searched_configs.kMinLargeAlloc;
    in >> searched_configs.kRoundLarge;

    for (size_t i = 0; i < allocatorConf::_GROUPS.size(); i++) {
        in >> allocatorConf::_GROUPS[i];
    }

    if (allocatorConf::_GROUPS[0] < std::numeric_limits<size_t>::max()) {
        group_enable_flag = true;
    }

    std::string line;
    while (std::getline(in, line)) {
        unique_hash_trace.emplace(line);
    }
    in.close();

    std::cout << searched_configs.kMinBlockSize << std::endl;
    std::cout << searched_configs.kSmallSize << std::endl;
    std::cout << searched_configs.kSmallBuffer << std::endl;
    std::cout << searched_configs.kLargeBuffer << std::endl;
    std::cout << searched_configs.kMinLargeAlloc << std::endl;
    std::cout << searched_configs.kRoundLarge << std::endl;
    for (auto htrace : unique_hash_trace) {
        std::cout << htrace << std::endl;
    }
}

void dump_opt_guidance(std::string filename) {
    std::ofstream out(filename);
    out << searched_configs.kMinBlockSize << std::endl;
    out << searched_configs.kSmallSize << std::endl;
    out << searched_configs.kSmallBuffer << std::endl;
    out << searched_configs.kLargeBuffer << std::endl;
    out << searched_configs.kMinLargeAlloc << std::endl;
    out << searched_configs.kRoundLarge << std::endl;

    for (size_t i = 0; i < allocatorConf::_GROUPS.size(); i++) {
        out << allocatorConf::_GROUPS[i] << std::endl;
    }

    for (auto stc : static_tensor_callpaths) {
        out << stc.first << std::endl;
        
        // print callpath, size, count for debugging
        std::cout << "static tensor callpath: " << stc.first
                     << " size: " << stc.second.first
                     << " count: " << stc.second.second << std::endl;
    }
    out.close();
}

void set_profiling_mode(bool mode) {
    is_profiling_mode = mode;
    if (!mode) {
        load_opt_guidance(dump_file_name);
    }
}

bool get_profiling_mode() {
    return is_profiling_mode;
}

/********************************************************************************
 ******************** Function definitions of allocatorMgr **********************
********************************************************************************/

allocatorMgr::allocatorMgr() : allocatorMgr(0, 0) {
}

allocatorMgr::allocatorMgr(int device, int stream) {
    this->device = device;
    this->stream = stream;

    // sanitizer_callbacks_subscribe();

}

allocatorMgr::~allocatorMgr() {
    std::cout << "Simulator destructor called" << std::endl;
    std::cout << "Simulator max reserved size: " << get_reserved_bytes() << std::endl;
    std::cout << "Simulator max allocated size: " << get_allocated_bytes() << std::endl;
    
    // sanitizer_callbacks_unsubscribe();
}

void allocatorMgr::test_simulator() {
    process_trace();
    auto memory_usage = simulate_allocator();
    std::cout << "Max reserved size: " << memory_usage << std::endl << std::endl;
    search_config_with_group();
}

bool allocatorMgr::check_constraints() {
    if (allocatorConf::get_kMinLargeAlloc() >= allocatorConf::get_kLargeBuffer()) {
        return false;
    }

    return true;
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
    // get the result without grouping
    evaluate_allocator(original_configs, original_configs);

    for (auto diff : GROUP_DIFFERENCES) {
        group_blocks(diff);
        if (evaluate_allocator(original_configs, original_configs)) {
            current_difference = diff;
            allocatorConf::BACKUP_GROUPS = allocatorConf::_GROUPS;
            // alloc_sim.set_group_enable_flag_sim(true);
            group_enable_flag = true;
        } else if(group_enable_flag) {
            // rollback
            allocatorConf::_GROUPS = allocatorConf::BACKUP_GROUPS;
        } else if (!group_enable_flag) {
            // disable group in sim
            alloc_sim.set_group_enable_flag_sim(false);
        }
        reset_allocator_memory_usage();
        empty_cache();
    }
    evaluate_allocator(original_configs, original_configs);
    log_configs(searched_configs);
    std::cout << "[allocatorMgr::search_group()]" << std::endl;
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
        std::cout << "[allocatorMgr::allocator_assert(bool expr)]" << std::endl;
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
    std::cout << "[allocatorMgr::search_config()]" << std::endl;
    report_configs(original_configs, searched_configs);

    // search_config_with_group();
}

// after search group
void allocatorMgr::search_config_with_group() {
    log_configs(original_configs);
    reset_allocator_memory_usage();
    empty_cache();
    searched_configs = original_configs;
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
                            // get the result without grouping
                            if (evaluate_allocator(searched_configs, prev_conf)) {
                                prev_conf = searched_configs;
                            }
                            reset_allocator_memory_usage();
                            empty_cache();

                            for (auto diff : GROUP_DIFFERENCES) {
                                group_blocks(diff);
                                if (evaluate_allocator(searched_configs, prev_conf)) {
                                    prev_conf = searched_configs;
                                    current_difference = diff;
                                    allocatorConf::BACKUP_GROUPS = allocatorConf::_GROUPS;
                                    // alloc_sim.set_group_enable_flag_sim(true);
                                    group_enable_flag = true;
                                } else if(group_enable_flag) {
                                    // rollback
                                    allocatorConf::_GROUPS = allocatorConf::BACKUP_GROUPS;
                                } else if (!group_enable_flag) {
                                    alloc_sim.set_group_enable_flag_sim(false);
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
    std::cout << "[allocatorMgr::search_config_with_group()]" << std::endl;
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
        allocated_size,
        reserved_size
    );
}

// check the functionality of the simulator by synchronously running it
void allocatorMgr::collect_trace_sync(void* ptr, int64_t size, bool real) {
    if (size > 0) {  // malloc
        Block* block = this->alloc_sim.malloc(this->device, size, this->stream);
        free_blocks.emplace(reinterpret_cast<uint64_t>(ptr), block);
    } else {  // free
        if (real) { // release the block
            return ;
        }
        auto block = free_blocks[reinterpret_cast<uint64_t>(ptr)];
        this->alloc_sim.free(block);
        free_blocks.erase(reinterpret_cast<uint64_t>(ptr));
    }
    increase_global_op_id();
}

void allocatorMgr::process_empty_cache_api() {
    if (enable_sync_collect_trace) {
        empty_cache();
    } else {
        // collect emtpy cache event
    }
}

void allocatorMgr::collect_api(AllocatorAPIType_t api_type) {
    if (api_type == ALLOCATOR_EMPYT_CACHE_API) {
        process_empty_cache_api();
    }
    increase_global_op_id();
}

// For functionality test
void allocatorMgr::functionality_test() {
    if (!_active_blocks.empty()) {
        for (auto b : _active_blocks) {
            _block_trace.emplace(b.second.first, std::make_pair(get_global_op_id(), b.second.second));
            increase_global_op_id();
        }
    }

    std::cout << "before reserved size: " << get_reserved_bytes() << std::endl;
    std::cout << "before allocated size: " << get_allocated_bytes() << std::endl;
    process_trace();
    simulate_allocator();
    std::cout << "after reserved size: " << get_reserved_bytes() << std::endl;
    std::cout << "after allocated size: " << get_allocated_bytes() << std::endl;
    
}

void allocatorMgr::collect_trace_async(void* ptr, int64_t size, bool real) {

}

void allocatorMgr::collect_trace(void* ptr, int64_t size, bool real) {
    if (enable_sync_collect_trace) {
        collect_trace_sync(ptr, size, real);
    } else {
        collect_trace_async(ptr, size, real);
    }
}

void allocatorMgr::collect_trace_stale(void* ptr, int64_t size) {
    if (size > 0) {  // malloc
        if (get_profiling_mode()) {
            auto callpath = get_callpath_hash();
            ptr2callpath.emplace(ptr, callpath);

            // static tensor analysis
            if (reclaimed_callpaths.find(callpath) == reclaimed_callpaths.end()) {
                auto static_tensor_cp = static_tensor_callpaths.find(callpath);
                if (iteration == 0) {
                    if (static_tensor_cp == static_tensor_callpaths.end()) {
                        static_tensor_callpaths.emplace(callpath, std::make_pair(size, 1));
                    } else {
                        static_tensor_cp->second.first += size;
                        static_tensor_cp->second.second += 1;
                    }
                } else {
                    if (static_tensor_cp != static_tensor_callpaths.end()) {
                        static_tensor_cp->second.first += size;
                        static_tensor_cp->second.second += 1;
                    }
                }
            }
        }
        _active_blocks.emplace(ptr, std::make_pair(get_global_op_id(), size));
    } else {  // free
        if (get_profiling_mode()) {
            auto callpath = ptr2callpath.find(ptr);
            if (callpath != ptr2callpath.end()) {
                auto static_tensor_cp = static_tensor_callpaths.find(callpath->second);
                if (static_tensor_cp != static_tensor_callpaths.end()) {
                    static_tensor_callpaths.erase(static_tensor_cp);
                }
                reclaimed_callpaths.emplace(callpath->second);
                ptr2callpath.erase(callpath);
            } 
        }
        auto b = _active_blocks.find(ptr);
        _block_trace.emplace(b->second.first, std::make_pair(get_global_op_id(), b->second.second));
        _active_blocks.erase(b);
    }
    increase_global_op_id();
}

char* allocatorMgr::malloc_cpu_memory_chunk(size_t size) {
    char* pointer = (char*) malloc(size);
    return pointer;
}

void allocatorMgr::free_cpu_memory_chunk(char* pointer) {
    free((void*)pointer);
}

bool allocatorMgr::iter_end(bool begin, size_t active_size) {
    bool result = false;
    if (get_profiling_mode()) {
        if (iteration == 0) { // search configs after the first iteration
            process_trace();
            current_reserved_size = simulate_allocator();
            search_config_with_group();
        }

        size_t max_monitored_iterations = 2;
        if (iteration == max_monitored_iterations) {
            // dump configs
            dump_opt_guidance(dump_file_name);
        }
    }
    else {
        if (false) { // change to optimize at beginning, always false for now
            process_trace();
            current_reserved_size = simulate_allocator();
            // search_config();
            // search_group();
            search_config_with_group();
            result = true;
            _active_blocks.clear();
        }
    }

    _block_trace.clear();
    iteration++;
    return result;
}

bool allocatorMgr::iteration_trigger(bool begin, size_t active_size) {
    size_t result = false;
    if (begin) {
        // do something
    } else {
        result = iter_end(begin, active_size);
    }
    return result;
}

void allocatorMgr::process_trace() {
    for (auto t : _block_trace) {
        op_id_map.emplace(t.first, true);
        op_id_map.emplace(t.second.first, false);
    }
}

size_t allocatorMgr::simulate_allocator() {
    for (auto op : op_id_map) {
        if (op.second) {
            auto orig_size = _block_trace[op.first].second;
            auto block = this->alloc_sim.malloc(this->device, orig_size, this->stream);
            free_blocks.emplace(_block_trace[op.first].first, block);
        } else {
            this->alloc_sim.free(free_blocks[op.first]);
        }
    }
    free_blocks.clear();

    auto reserved_size = get_reserved_bytes();
    auto allocated_size = get_allocated_bytes();

    log_configs(searched_configs);
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
    if (group_enable_flag) {
        std::cout << "Group difference: " << current_difference << std::endl;
        for (auto g : allocatorConf::_GROUPS) {
            std::cout << g << " ";
        }
        std::cout << std::endl;
    }
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
    for (auto t : _block_trace) {
        if (t.second.second > allocatorConf::get_kLargeBuffer()) {
            block_sizes.insert(t.second.second);
        }
    }

    if (block_sizes.empty()) {
        return;
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
    if (group_boundary != *block_sizes.rbegin()) {
        allocatorConf::_GROUPS[index] = *block_sizes.rbegin();
    }
    alloc_sim.set_group_enable_flag_sim(true);
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

std::string allocatorMgr::get_callpath_hash() {
    auto python_states = get_python_states();
    auto cpp_callpath = get_backtrace();
    return sha256(python_states + cpp_callpath);
    // return python_states + cpp_callpath;
}

std::string allocatorMgr::get_python_states() {
    size_t num_states = 0;

    python_state_get(MAX_NUM_STATES, python_states, &num_states);

    std::stringstream ss;

    for (size_t i = 0; i < num_states; i++) {
        ss << std::string(python_states[i].file_name) << ":"
           << std::to_string(python_states[i].lineno) << std::endl
           << std::string(python_states[i].function_name) << ":"
           << std::to_string(python_states[i].function_first_lineno) << std::endl;
}
    // std::cout << ss.str() << std::endl;
    return ss.str();
}

bool allocatorMgr::check_callpath() {
    if (iteration == 0 && !get_profiling_mode()) {
        auto callpath_hash = get_callpath_hash();
        auto it = unique_hash_trace.find(callpath_hash);
        if (it != unique_hash_trace.end()) {
            return true;
        }
    }
    return false;
}

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim
