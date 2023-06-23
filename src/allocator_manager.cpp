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

    if (sim_control::SimulatorModeController::is_group_optimization()) {
        for (size_t i = 0; i < allocatorConf::_GROUPS.size(); i++) {
            in >> allocatorConf::_GROUPS[i];
        }

        if (allocatorConf::_GROUPS[0] < std::numeric_limits<size_t>::max()) {
            group_enable_flag = true;
        }
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

    if (sim_control::SimulatorModeController::is_group_optimization()) {
        for (size_t i = 0; i < allocatorConf::_GROUPS.size(); i++) {
            out << allocatorConf::_GROUPS[i] << std::endl;
        }
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
    sim_control::SimulatorModeController::set_profiling(mode);
    if (!mode) {
        load_opt_guidance(dump_file_name);
    }
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

    sim_control::SimulatorModeController::init();

    sim_control::SimulatorModeController::show();

    DumpDebugging::enableDumppingDebugInfo();

    if (!sim_control::SimulatorModeController::is_profiling() && !sim_control::SimulatorModeController::is_functionality_checking()) {
        // load_opt_guidance(dump_file_name);
        apply_configs(searched_configs);
    }
}

allocatorMgr::~allocatorMgr() {
    std::cout << "Simulator destructor called" << std::endl;
    std::cout << "Simulator max reserved size: " << get_max_reserved_bytes() << std::endl;
    std::cout << "Simulator max allocated size: " << get_max_allocated_bytes() << std::endl;
    
    // sanitizer_callbacks_unsubscribe();

    if (sim_control::SimulatorModeController::is_async_tracing() && sim_control::SimulatorModeController::is_functionality_checking()) {
        test_functionality_under_collect_trace_async();
    }

    if (sim_control::SimulatorModeController::is_profiling() && sim_control::SimulatorModeController::is_functionality_checking()) {
        // may not used, mark it as deprecated
        optimize_functionality();
    }

    if (sim_control::SimulatorModeController::is_trace_dumpping()) {
        dump_memory_usage_to_file();
    }
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
        std::cout << "[allocatorMgr::allocator_assert(bool expr)!!!]" << std::endl;
        report_configs(original_configs, searched_configs);
        exit(1);
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
        reserved_size = get_max_reserved_bytes();
        allocated_size = get_max_allocated_bytes();
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

void allocatorMgr::process_empty_cache_api() {
    if (!sim_control::SimulatorModeController::is_async_tracing()) {
        empty_cache();
    } else {
        // collect emtpy cache event
        _api_trace.emplace(get_global_op_id(), ALLOCATOR_EMPYT_CACHE);
    }
}

void allocatorMgr::collect_api(AllocatorEventType_t api_type) {
    if (api_type == ALLOCATOR_EMPYT_CACHE) {
        process_empty_cache_api();
    }
    increase_global_op_id();
}

// For functionality test
void allocatorMgr::test_functionality_under_collect_trace_async() {
    if (!_active_blocks.empty()) {
        for (auto b : _active_blocks) {
            _block_trace.emplace(b.second.first, std::make_pair(get_global_op_id(), b.second.second));
            increase_global_op_id();
        }
    }

    std::cout << "Before functionality check max reserved size: " << get_max_reserved_bytes() << std::endl;
    std::cout << "Before functionality check max allocated size: " << get_max_allocated_bytes() << std::endl;
    process_trace();
    simulate_allocator();
    std::cout << "After functionality check max reserved size: " << get_max_reserved_bytes() << std::endl;
    std::cout << "After functionality check max allocated size: " << get_max_allocated_bytes() << std::endl;
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

// check the simulation functionality of the simulator by asynchronously running it (run after trace collection)
void allocatorMgr::collect_trace_async(void* ptr, int64_t size, bool real) {
    if (size > 0) {  // malloc
        _active_blocks.emplace(ptr, std::make_pair(get_global_op_id(), size));
    } else {  // free
        if (real) { // release the block
            return ;
        }
        auto b = _active_blocks.find(ptr);
        _block_trace.emplace(b->second.first, std::make_pair(get_global_op_id(), b->second.second));
        _active_blocks.erase(b);
    }
    increase_global_op_id();
}

void allocatorMgr::collect_trace(void* ptr, int64_t size, bool real) {
    if (!sim_control::SimulatorModeController::is_async_tracing()) {
        collect_trace_sync(ptr, size, real);
    } else {
        if (sim_control::SimulatorModeController::is_functionality_checking()) {
            collect_trace_async(ptr, size, real);
        } else {
            collect_trace_opt2(ptr, size, real);
        }
    }
}

void allocatorMgr::optimize_functionality() {
    if (!_active_blocks.empty()) {
        for (auto b : _active_blocks) {
            _block_trace.emplace(b.second.first, std::make_pair(get_global_op_id(), b.second.second));
            increase_global_op_id();
        }
    }

    process_trace();
    current_reserved_size = simulate_allocator();
    search_config();

    dump_opt_guidance(dump_file_name);
}

void allocatorMgr::collect_trace_opt(void* ptr, int64_t size, bool real) {
    if (size > 0) {  // malloc
        _active_blocks.emplace(ptr, std::make_pair(get_global_op_id(), size));

    } else {  // free
        if (real) { // release the block
            return ;
        }
        auto b = _active_blocks.find(ptr);
        _block_trace.emplace(b->second.first, std::make_pair(get_global_op_id(), b->second.second));
        _active_blocks.erase(b);
    }
    increase_global_op_id();
}

// @Lin-Mao(todo): a issue to be investigated: the gap of dynamic tensors simulation even larger than the original one
void allocatorMgr::collect_trace_opt2(void* ptr, int64_t size, bool real) {
    if (size > 0) {  // malloc
        if (sim_control::SimulatorModeController::is_profiling()) {
            auto callpath = get_callpath_hash();
            ptr2callpath.emplace(ptr, callpath);

            // static tensor analysis
            // static tensors is the tensors that are at first iteration and never reclaimed in the later iterations
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
        if (real) {
            return ; // do nothing for real free, handled by empty_cache
        }
        if (sim_control::SimulatorModeController::is_profiling()) {
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

void allocatorMgr::dump_trace_to_file() {
    std::string dump_path = "./output/";
    std::string trace_file = "trace.csv";

    // in case the directory is not existed
    if (!fs::is_directory(dump_path)) {
        int ret = system(("mkdir -p " + dump_path).c_str());
        if (ret != 0) {}
    }

    if (iteration == 0) {
        std::ofstream output(dump_path + trace_file);
        output.close();
    }

    // max iterations to collect trace
    size_t max_monitored_iterations = 2;
    if (iteration > max_monitored_iterations) {
        return ;
    }

    std::ofstream output(dump_path + trace_file, std::ios::app);
    output << "iteration: " << iteration << ", block trace" << std::endl;
    for (auto b : _block_trace) {
        // malloc_op_id, free_op_id, size
        output << b.first << "," << b.second.first << "," << b.second.second << std::endl;
    }
    output << "iteration: " << iteration << ", active block trace" << std::endl;
    for (auto b : _active_blocks) {
        // ptr, malloc_op_id, size
        output << reinterpret_cast<uint64_t>(b.first) << "," << b.second.first << "," << b.second.second << std::endl;
    }
    output << "iteration: " << iteration << ", api trace" << std::endl;
    for (auto a : _api_trace) {
        // api_op_id, api_type
        output << a.first << "," << static_cast<uint32_t>(a.second) << std::endl;
    }
    output.close();

    _block_trace.clear();
    _api_trace.clear();

}

bool allocatorMgr::iter_end() {
    bool result = false;    // indicates whether applying a online optimization
    if (sim_control::SimulatorModeController::is_trace_dumpping()) {
        dump_trace_to_file();
        iteration++;
        return result;
    }
    if (sim_control::SimulatorModeController::is_profiling()) {
        size_t max_monitored_iterations = 2;
        // search configs when reaching the max monitored iterations
        if (iteration == max_monitored_iterations) {
            process_trace();
            current_reserved_size = simulate_allocator();
            std::cout << "init_reserved_size: " << current_reserved_size << std::endl;
            if (sim_control::SimulatorModeController::is_config_optimization()) {
                search_config();
            } else if (sim_control::SimulatorModeController::is_group_optimization()) {
                search_config_with_group();
            }
            // dump configs
            dump_opt_guidance(dump_file_name);
        }
    }
    else {
        if (false) { // change to optimize at program beginning, always false for now
            process_trace();
            current_reserved_size = simulate_allocator();
            // search_config();
            // search_group();
            search_config_with_group();
            result = true;
            _active_blocks.clear();
        }
    }

    // ?: continuous profiling should clear trace after each iteration
    // _block_trace.clear();
    iteration++;
    return result;
}

bool allocatorMgr::iteration_trigger(bool begin) {
    size_t result = false;
    if (begin) {
        // do something
    } else {
        result = iter_end();
    }
    return result;
}

void allocatorMgr::process_trace() {
    // determine if process the active blocks
    if (!sim_control::SimulatorModeController::is_static_tensor_analysis()) {
        if (!_active_blocks.empty()) {
            for (auto b : _active_blocks) {
                _block_trace.emplace(b.second.first, std::make_pair(get_global_op_id(), b.second.second));
            }
            increase_global_op_id();
        }
    }
    for (auto t : _block_trace) {
        opid2event.emplace(t.first, ALLOCATOR_MALLOC_BLOCK);
        opid2event.emplace(t.second.first, ALLOCATOR_FREE_BLOCK);
    }

    for (auto t : _api_trace) {
        opid2event.emplace(t.first, t.second);
    }
}

size_t allocatorMgr::simulate_allocator() {
    for (auto op : opid2event) {
        if (op.second == ALLOCATOR_MALLOC_BLOCK) {
            auto orig_size = _block_trace[op.first].second;
            auto block = this->alloc_sim.malloc(this->device, orig_size, this->stream);
            free_blocks.emplace(_block_trace[op.first].first, block);
        } else if (op.second == ALLOCATOR_FREE_BLOCK) {
            this->alloc_sim.free(free_blocks[op.first]);
        } else if (op.second == ALLOCATOR_EMPYT_CACHE) {
            empty_cache();
        }
    }
    free_blocks.clear();

    auto reserved_size = get_max_reserved_bytes();
    auto allocated_size = get_max_allocated_bytes();

    log_configs(searched_configs);
    allocator_assert(reserved_size >= allocated_size);
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
    alloc_sim.empty_cache();
}

std::pair<size_t, size_t> allocatorMgr::get_allocator_memory_usage() {
    return alloc_sim.get_max_memory_usage();
}

size_t allocatorMgr::get_max_reserved_bytes() {
    return alloc_sim.get_max_reserved_bytes();
}

size_t allocatorMgr::get_max_allocated_bytes() {
    return alloc_sim.get_max_allocated_bytes();
}

void allocatorMgr::reset_allocator_memory_usage() {
    alloc_sim.reset_memory_usage();
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
    if (iteration == 0 && !sim_control::SimulatorModeController::is_profiling()) {
        auto callpath_hash = get_callpath_hash();
        auto it = unique_hash_trace.find(callpath_hash);
        if (it != unique_hash_trace.end()) {
            return true;
        }
    }
    return false;
}

void allocatorMgr::collect_memory_usage(int64_t size, size_t allocated_cur, size_t reserved_cur) {
    max_allocator_allocated = std::max(max_allocator_allocated, allocated_cur);
    max_allocator_reserved = std::max(max_allocator_reserved, reserved_cur);
    auto mem = std::make_tuple(get_global_op_id(), size, allocated_cur, reserved_cur);
    _allocator_mem_usage.push_back(mem);
}

void allocatorMgr::dump_memory_usage_to_file() {
    std::string dump_path = "./output/";
    std::string trace_file = "memory.csv";

    // in case the directory is not existed
    if (!fs::is_directory(dump_path)) {
        int ret = system(("mkdir -p " + dump_path).c_str());
        if (ret != 0) {}
    }

    std::ofstream output(dump_path + trace_file);
    for (auto m : _allocator_mem_usage) {
        output << std::get<0>(m) << "," << std::get<1>(m) << "," << std::get<2>(m) << "," << std::get<3>(m) << std::endl;
    }
    output << std::endl;
    output << "max_allocated," << max_allocator_allocated << std::endl;
    output << "max_reserved," << max_allocator_reserved << std::endl;
    output.close();
}


}  // namespace AllocatorSim
}  // namespace cuda
}  // namespace c10
