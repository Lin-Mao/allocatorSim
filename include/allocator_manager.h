#ifndef ALLOCATOR_MANAGER_H
#define ALLOCATOR_MANAGER_H

#include "allocator_simulator.h"

namespace c10 {
namespace cuda {
namespace AllocatorSim {

struct Configs {
    size_t kMinBlockSize;
    size_t kSmallSize;
    size_t kSmallBuffer;
    size_t kLargeBuffer;
    size_t kMinLargeAlloc;
    size_t kRoundLarge;

    size_t m_max_split_size;
    size_t m_roundup_power2_divisions;
    size_t m_roundup_bypass_threshold;
    double m_garbage_collection_threshold;
    uint64_t m_memory_segment_address_start;
    uint64_t m_memory_segment_address_interval;

    size_t allocated_size;
    size_t reserved_size;

    Configs() = default;

    Configs (
        size_t kMinBlockSize,
        size_t kSmallSize,
        size_t kSmallBuffer,
        size_t kLargeBuffer,
        size_t kMinLargeAlloc,
        size_t kRoundLarge,
        size_t m_max_split_size,
        size_t m_roundup_power2_divisions,
        size_t m_roundup_bypass_threshold,
        double m_garbage_collection_threshold,
        uint64_t m_memory_segment_address_start,
        uint64_t m_memory_segment_address_interval,
        size_t allocated_size,
        size_t reserved_size)
          : kMinBlockSize(kMinBlockSize),
            kSmallSize(kSmallSize),
            kSmallBuffer(kSmallBuffer),
            kLargeBuffer(kLargeBuffer),
            kMinLargeAlloc(kMinLargeAlloc),
            kRoundLarge(kRoundLarge),

            m_max_split_size(m_max_split_size),
            m_roundup_power2_divisions(m_roundup_power2_divisions),
            m_roundup_bypass_threshold(m_roundup_bypass_threshold),
            m_garbage_collection_threshold(m_garbage_collection_threshold),
            m_memory_segment_address_start(m_memory_segment_address_start),
            m_memory_segment_address_interval(m_memory_segment_address_interval),
            
            allocated_size(allocated_size),
            reserved_size(reserved_size) {}

    Configs (
        size_t kMinBlockSize,
        size_t kSmallSize,
        size_t kSmallBuffer,
        size_t kLargeBuffer,
        size_t kMinLargeAlloc,
        size_t kRoundLarge,
        size_t allocated_size,
        size_t reserved_size)
          : Configs(kMinBlockSize, kSmallSize, kSmallBuffer, kLargeBuffer,
            kMinLargeAlloc, kRoundLarge, 0, 0, 0, 0.0, 0, 0, allocated_size, reserved_size) {}
};

typedef enum AllocatorEventType {
    ALLOCATOR_MALLOC_BLOCK = 0,
    ALLOCATOR_FREE_BLOCK = 1,
    ALLOCATOR_MALLOC_SEGMENT = 2,
    ALLOCATOR_RELEASE_SEGMENT = 3,
    ALLOCATOR_EMPYT_CACHE = 4,
    NUMS_OF_ALLOCATOR_EVENT = 5
} AllocatorEventType_t;

// For torch.cuda.enable_profiling()
void set_profiling_mode(bool mode);

class allocatorMgr {
private:
    int device;
    int stream;
    allocatorSim alloc_sim;
    

    size_t current_reserved_size = std::numeric_limits<size_t>::max();
    
    // <ptr, <op_id, size>>
    std::map<void*, std::pair<op_id_t, size_t>> _active_blocks;
    trace_t _block_trace;
    std::map<op_id_t, AllocatorEventType_t> _api_trace;

    // <op_id, <size, allocated_curr, reserved_curr>>
    std::vector<std::tuple<op_id_t, int64_t, size_t, size_t>> _allocator_mem_usage;
    size_t max_allocator_reserved = 0;
    size_t max_allocator_allocated = 0;

    // not used, need to collect alloc_size if used
    // std::map<void*, std::pair<uint64_t, size_t>> _active_segments;
    // trace_t _segment_trace;

    // <op_id, events>
    std::map<op_id_t, AllocatorEventType_t> opid2event;
    std::unordered_map<op_id_t, Block*> free_blocks;

    // may not be used
    std::unordered_map<void*, uint64_t> realptr2simptr;


    const std::set<size_t> kMinBlockSize_candidates {256, 512, 1024, 2048, 4096};
    const std::set<size_t> kSmallSize_candidates {1048576/2, 1048576, 1048576*3/2, 1048576*2};
    const std::set<size_t> kSmallBuffer_candidates {2097152, 2097152*2, 2097152*3, 2097152*4, 2097152*5};
    const std::set<size_t> kLargeBuffer_candidates {20971520/2, 20971520, 20971520*3/2, 20971520*2, 20971520*5/2};
    const std::set<size_t> kMinLargeAlloc_candidates {10485760*2, 10485760*4, 10485760*6, 10485760*8, 10485760*10};
    const std::set<size_t> kRoundLarge_candidates {2097152, 2097152*2, 2097152*4, 2097152*8, 2097152*10, 2097152*12};
    const std::set<float> GROUP_DIFFERENCES {0.2, 0.6, 1.2, 1.6, 2.0};

    std::array<std::set<size_t>, CONFIG_NUMS> ALL_CANDIDATES = {
        kMinBlockSize_candidates, kSmallSize_candidates, kSmallBuffer_candidates,
        kLargeBuffer_candidates, kMinLargeAlloc_candidates, kRoundLarge_candidates
    };

    float current_difference = 0.0;

private:

    bool check_constraints();

    bool check_configs(Configs& config);

    void log_configs(Configs& configs, bool get_mem = true);

    void apply_configs(const Configs& configs);

    void report_configs(const Configs& conf1, const Configs& conf2);

    // true means new config works
    bool evaluate_allocator(Configs configs, Configs prev_conf);

    void allocator_assert(bool expr);

    void process_trace();

    size_t simulate_allocator();

    void search_config();

    void search_group();

    void search_config_with_group();

    void group_blocks(const float& difference);

    bool iter_end();

    std::string get_callpath_hash();
    
    std::string get_python_states();

    size_t get_max_reserved_bytes();

    size_t get_max_allocated_bytes();

    std::pair<size_t, size_t> get_allocator_memory_usage();

    void reset_allocator_memory_usage();

    void show_allocator_memory_usage();

    size_t get_grouped_allocation_size(size_t size);

    void process_empty_cache_api();

    void test_functionality_under_collect_trace_async();

    // real is to determine if it's a real (de)allocation
    void collect_trace_sync(void* ptr, int64_t size, bool real = false);

    void collect_trace_async(void* ptr, int64_t size, bool real = false);

    void collect_trace_opt(void* ptr, int64_t size, bool real = false);

    void collect_trace_opt2(void* ptr, int64_t size, bool real = false);

    void optimize_functionality();

    void empty_cache();

    // have not access to BD models, use this func to collect the model trace
    void dump_trace_to_file();

    void dump_memory_usage_to_file();
    
public:
    allocatorMgr();

    allocatorMgr(int device, int stream);

    ~allocatorMgr();

    void test_simulator();

    void collect_trace(void* ptr, int64_t size, bool real = false);

    void collect_api(AllocatorEventType_t api_type);

    bool iteration_trigger(bool begin = true);

    char* malloc_cpu_memory_chunk(size_t size);

    void free_cpu_memory_chunk(char* pointer);

    size_t get_allocation_size(size_t size);

    bool check_callpath();

    void collect_memory_usage(int64_t size, size_t allocated_cur, size_t reserved_cur);

};

}  // namespace AllocatorSim
}  // namespace cuda
}  // namespace c10

#endif  // ALLOCATOR_MANAGER_H
