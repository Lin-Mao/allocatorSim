/**
 * Configurations of allocator.
*/
#ifndef ALLOCATOR_CFG_H
#define ALLOCATOR_CFG_H

#include <cstddef>
#include <limits>
#include <cstdint>

// all sizes are rounded to at least 512 bytes
constexpr size_t kMinBlockSize =512;
// largest "small" allocation is 1 MiB
constexpr size_t kSmallSize = 1048576;
// "small" allocations are packed in 2 MiB blocks
constexpr size_t kSmallBuffer = 2097152;
// "large" allocations may be packed in 20 MiB blocks
constexpr size_t kLargeBuffer = 20971520;
// allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kMinLargeAlloc = 10485760;
// round up large allocations to 2 MiB
constexpr size_t kRoundLarge = 2097152;

class allocatorConf {
private:
    static size_t m_max_split_size;
    static size_t m_roundup_power2_divisions;
    static size_t m_roundup_bypass_threshold;
    static double m_garbage_collection_threshold;
    static uint64_t m_memory_segment_address_start;
    static uint64_t m_memory_segment_address_interval;

public:
    static size_t get_max_split_size();

    static size_t get_roundup_power2_divisions();

    static size_t get_roundup_bypass_threshold();

    static double get_garbage_collection_threshold();

    static uint64_t get_memory_segment_address_start();

    static uint64_t get_memory_segment_address_interval();

};

#endif // ALLOCATOR_CFG_H