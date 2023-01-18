#include "allocator_conf.h"

namespace c10 {
namespace cuda {
namespace AllocatorSim {

size_t allocatorConf::kMinBlockSize =512;
// largest "small" allocation is 1 MiB
size_t allocatorConf::kSmallSize = 1048576;
// "small" allocations are packed in 2 MiB blocks
size_t allocatorConf::kSmallBuffer = 2097152;
// "large" allocations may be packed in 20 MiB blocks
size_t allocatorConf::kLargeBuffer = 20971520;
// allocations between 1 and 10 MiB may use kLargeBuffer
size_t allocatorConf::kMinLargeAlloc = 10485760;
// round up large allocations to 2 MiB
size_t allocatorConf::kRoundLarge = 2097152;

std::array<SET_FUNC, CONFIG_NUMS> allocatorConf::set_funcs = {
    set_kMinBlockSize, set_kSmallSize, set_kSmallBuffer,
    set_kLargeBuffer, set_kMinLargeAlloc, set_kRoundLarge
};
std::array<GET_FUNC, CONFIG_NUMS> allocatorConf::get_funcs = {
    get_kMinBlockSize, get_kSmallSize, get_kSmallBuffer,
    get_kLargeBuffer, get_kMinLargeAlloc, get_kRoundLarge
};

size_t allocatorConf::m_max_split_size = std::numeric_limits<size_t>::max();

size_t allocatorConf::m_roundup_power2_divisions = 0;

size_t allocatorConf::m_roundup_bypass_threshold = std::numeric_limits<size_t>::max();

double allocatorConf::m_garbage_collection_threshold = 0;

uint64_t allocatorConf::m_memory_segment_address_start = 1000;

uint64_t allocatorConf::m_memory_segment_address_interval = 1000;

size_t allocatorConf::get_kMinBlockSize() {
    return kMinBlockSize;
}

void allocatorConf::set_kMinBlockSize(size_t size) {
    kMinBlockSize = size;
}

size_t allocatorConf::get_kSmallSize() {
    return kSmallSize;
}

void allocatorConf::set_kSmallSize(size_t size) {
    kSmallSize = size;
}

size_t allocatorConf::get_kSmallBuffer() {
    return kSmallBuffer;
}

void allocatorConf::set_kSmallBuffer(size_t size) {
    kSmallBuffer = size;
}

size_t allocatorConf::get_kLargeBuffer() {
    return kLargeBuffer;
}

void allocatorConf::set_kLargeBuffer(size_t size) {
    kLargeBuffer = size;
}

size_t allocatorConf::get_kMinLargeAlloc() {
    return kMinLargeAlloc;
}

void allocatorConf::set_kMinLargeAlloc(size_t size) {
    kMinLargeAlloc = size;
}

size_t allocatorConf::get_kRoundLarge() {
    return kRoundLarge;
}

void allocatorConf::set_kRoundLarge(size_t size) {
    kRoundLarge = size;
}

size_t allocatorConf::get_max_split_size() {
    return m_max_split_size;
}

void allocatorConf::set_max_split_size(size_t size) {
    m_max_split_size = size;
}

size_t allocatorConf::get_roundup_power2_divisions() {
    return m_roundup_power2_divisions;
}

void allocatorConf::set_roundup_power2_divisions(size_t val) {
    m_roundup_power2_divisions = val;
}

size_t allocatorConf::get_roundup_bypass_threshold() {
    return m_roundup_bypass_threshold;
}

void allocatorConf::set_roundup_bypass_threshold(size_t threshold) {
    m_roundup_bypass_threshold = threshold;
}

double allocatorConf::get_garbage_collection_threshold() {
    return m_garbage_collection_threshold;
}

void allocatorConf::set_garbage_collection_threshold(double threshold) {
    m_garbage_collection_threshold = threshold;
}

uint64_t allocatorConf::get_memory_segment_address_start() {
    return m_memory_segment_address_start;
}

void allocatorConf::set_memory_segment_address_start(uint64_t start) {
    m_memory_segment_address_start = start;
}

uint64_t allocatorConf::get_memory_segment_address_interval() {
    return m_memory_segment_address_interval;
}

void allocatorConf::set_memory_segment_address_interval(uint64_t interval) {
    m_memory_segment_address_interval = interval;
}

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim