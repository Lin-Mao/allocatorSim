#ifndef ALLOCATOR_MGR_H
#define ALLOCATOR_MGR_H

#include "allocator_utils.h"
#include "allocator_sim.h"

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

class allocatorMgr {
private:
    int device;
    int stream;
    allocatorSim alloc_sim;
    std::map<Block*, size_t> block_ref_map;
    
    uint64_t op_id = 0;
    size_t current_reserved_size;
    std::map<void*, std::pair<uint64_t, size_t>> _active_blocks;
    blockMap_t _trace;
    Configs original_configs;
    Configs searched_configs;

    std::set<size_t> kMinBlockSize_candidates {256, 512, 1024, 2048, 4096};
    std::set<size_t> kSmallSize_candidates {1048576/2, 1048576, 1048576*3/2};
    std::set<size_t> kSmallBuffer_candidates {20971520, 20971520*2, 20971520*3, 20971520*4, 20971520*5};
    std::set<size_t> kLargeBuffer_candidates {20971520/2, 20971520, 20971520*3/2, 20971520*2, 20971520*5/2, 20971520*3};
    std::set<size_t> kMinLargeAlloc_candidates {10485760*2, 10485760*4, 10485760*6, 10485760*8, 10485760*10};
    std::set<size_t> kRoundLarge_candidates {2097152, 2097152*2, 2097152*4, 2097152*8, 2097152*16, 2097152*24, 2097152*32};

private:
    template<typename FUNC1, typename FUNC2, typename candidate_t>
    void search_candidates(FUNC1 get_func, FUNC2 set_func, std::set<candidate_t> candidates);

    bool check_constraints();

    void log_configs(Configs& configs);

    void malloc_block(size_t orig_size, size_t ref);

    void update_block_reference();

    void free_block();
    
public:
    allocatorMgr();

    allocatorMgr(int device, int stream);

    void empty_cache();

    std::pair<size_t, size_t> get_allocator_memory_usage();

    void reset_allocator_memory_usage();

    void show_allocator_memory_usage();

    std::pair<size_t, size_t> simulate_allocator();

    void collector_trace(void* ptr, int64_t size);

    void search_configs();

    void report_config();

};

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim

#endif  // ALLOCATOR_MGR_H