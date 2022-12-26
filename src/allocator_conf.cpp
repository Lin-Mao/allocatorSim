#include "allocator_conf.h"

size_t allocatorConf::m_max_split_size = std::numeric_limits<size_t>::max();

size_t allocatorConf::m_roundup_power2_divisions = 0;

size_t allocatorConf::m_roundup_bypass_threshold = std::numeric_limits<size_t>::max();

double allocatorConf::m_garbage_collection_threshold = 0;

uint64_t allocatorConf::m_memory_segment_address_start = 1000;

uint64_t allocatorConf::m_memory_segment_address_interval = 1000;


size_t allocatorConf::get_max_split_size() {
    return m_max_split_size;
}

size_t allocatorConf::get_roundup_power2_divisions() {
    return m_roundup_power2_divisions;
}

size_t allocatorConf::get_roundup_bypass_threshold() {
    return m_roundup_bypass_threshold;
}

double allocatorConf::get_garbage_collection_threshold() {
    return m_garbage_collection_threshold;
}

uint64_t allocatorConf::get_memory_segment_address_start() {
    return m_memory_segment_address_start;
}

uint64_t allocatorConf::get_memory_segment_address_interval() {
    return m_memory_segment_address_interval;
}