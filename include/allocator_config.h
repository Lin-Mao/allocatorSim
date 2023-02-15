/**
 * Configurations of allocator.
*/
#ifndef ALLOCATOR_CONFIG_H
#define ALLOCATOR_CONFIG_H

#include <cstddef>
#include <limits>
#include <cstdint>
#include <array>

namespace c10 {
namespace cuda {
namespace AllocatorSim {

#define CONFIG_NUMS 6
#define GROUP_NUMS 5

using SET_FUNC = void(*)(size_t);
using GET_FUNC = size_t(*)();

class allocatorConf {
private:
    static size_t kMinBlockSize;
    static size_t kSmallSize;
    static size_t kSmallBuffer;
    static size_t kLargeBuffer;
    static size_t kMinLargeAlloc;
    static size_t kRoundLarge;

    static size_t m_max_split_size;
    static size_t m_roundup_power2_divisions;
    static size_t m_roundup_bypass_threshold;
    static double m_garbage_collection_threshold;
    static uint64_t m_memory_segment_address_start;
    static uint64_t m_memory_segment_address_interval;

public:
    static std::array<SET_FUNC, CONFIG_NUMS> set_funcs;
    static std::array<GET_FUNC, CONFIG_NUMS> get_funcs;

    // pair: <boundary, size>
    static std::array<size_t, GROUP_NUMS> _GROUPS;

    static std::array<size_t, GROUP_NUMS> BACKUP_GROUPS;

    static size_t get_kMinBlockSize();

    static void set_kMinBlockSize(size_t size);

    static size_t get_kSmallSize();

    static void set_kSmallSize(size_t size);

    static size_t get_kSmallBuffer();

    static void set_kSmallBuffer(size_t size);

    static size_t get_kLargeBuffer();

    static void set_kLargeBuffer(size_t size);

    static size_t get_kMinLargeAlloc();
    
    static void set_kMinLargeAlloc(size_t size);

    static size_t get_kRoundLarge();

    static void set_kRoundLarge(size_t size);

    static size_t get_max_split_size();

    static void set_max_split_size(size_t size);

    static size_t get_roundup_power2_divisions();

    static void set_roundup_power2_divisions(size_t val);

    static size_t get_roundup_bypass_threshold();

    static void set_roundup_bypass_threshold(size_t threshold);

    static double get_garbage_collection_threshold();

    static void set_garbage_collection_threshold(double threshold);

    static uint64_t get_memory_segment_address_start();

    static void set_memory_segment_address_start(uint64_t start);

    static uint64_t get_memory_segment_address_interval();

    static void set_memory_segment_address_interval(uint64_t interval);

};

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim

#endif // ALLOCATOR_CONFIG_H
