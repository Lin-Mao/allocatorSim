/**
 * Configurations of allocator.
*/
#ifndef ALLOCATOR_CFG_H
#define ALLOCATOR_CFG_H

#include <cstddef>
#include <limits>

class allocatorConf {
private:
    static size_t m_max_split_size;
    static size_t m_roundup_power2_divisions;
    static size_t m_roundup_bypass_threshold;
    static double m_garbage_collection_threshold;
    static size_t m_memory_segment_address_start;
    static size_t m_memory_segment_address_interval;

public:
    static size_t get_max_split_size();

    static size_t get_roundup_power2_divisions();

    static size_t get_roundup_bypass_threshold();

    static double get_garbage_collection_threshold();

    static size_t get_memory_segment_address_start();

    static size_t get_memory_segment_address_interval();

};

#endif // ALLOCATOR_CFG_H