#ifndef ALLOCATOR_OPT_H
#define ALLOCATOR_OPT_H

#include "allocator_mgr.h"

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

class allocatorOpt {
private:
    allocatorMgr alloc_mgr;
    blockMap_t trace;
    uint64_t max;
    uint64_t min;
    size_t current_max_size;
    Configs original_configs;
    Configs searched_configs;

    std::set<size_t> kMinBlockSize_candidates {256, 512, 1024, 2048, 4096};
    std::set<size_t> kSmallSize_candidates {1048576/2, 1048576, 1048576*3/2};
    std::set<size_t> kSmallBuffer_candidates {20971520, 20971520*2, 20971520*3, 20971520*4, 20971520*5};
    std::set<size_t> kLargeBuffer_candidates {20971520/2, 20971520, 20971520*3/2, 20971520*2, 20971520*5/2, 20971520*3};
    std::set<size_t> kMinLargeAlloc_candidates {1048576*2, 10485760*4, 10485760*6, 10485760*8, 10485760*10};
    std::set<size_t> kRoundLarge_candidates {2097152, 2097152*2, 2097152*4, 2097152*8, 2097152*16};

private:
    void search_kMinBlockSize();
    
    void search_kSmallSize();

    void search_kSmallBuffer();

    void search_kLargeBuffer();

    void search_kMinLargeAlloc();

    void search_kRoundLarge();

    void log_configs(Configs& configs);

public:
    allocatorOpt(blockMap_t trace, uint64_t max, uint64_t min);

    void search_configs();

    std::pair<size_t, size_t> evaluate_model();

    void report_config();

};

#endif  // ALLOCATOR_OPT_H