/**
 * The states of allocator.
*/
#ifndef ALLOCATOR_PROFILER_H
#define ALLOCATOR_PROFILER_H

#include "allocator_utils.h"

#include <fstream>

#define DISABLE_ALLOCATOR_PROF 1

#if DISABLE_ALLOCATOR_PROF
#define ALLOCATOR_PROF_ENABLE() return
#else
#define ALLOCATOR_PROF_ENABLE() ((void)0)
#endif

namespace c10 {
namespace cuda {
namespace AllocatorSim {

struct Status;
struct OpInfo;
struct BlockInfo;
struct SegmentInfo;
struct MemoryRange;

struct Status {
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;

    Status() = default;

    Status(int64_t current, int64_t peak, int64_t allocated, int64_t freed)
        : current(current), peak(peak), allocated(allocated), freed(freed) {}

    Status(const Status& other)
        : Status(other.current, other.peak, other.allocated, other.freed) {}
};

struct OpInfo {
    uint64_t op_id;
    bool is_alloc;  // true: segment alloc, false: block alloc
    bool is_free;
    bool is_release;
    bool is_split;

    size_t allocated_size;
    size_t max_allocated_size;
    size_t reserved_size;
    size_t max_reserved_size;

    float utilization_ratio;
    float fragmentation;

    OpInfo() = default;

    OpInfo(
        uint64_t op_id,
        bool is_alloc,
        bool is_free,
        bool is_release,
        bool is_split,
        size_t allocated_size,
        size_t max_allocated_size,
        size_t reserved_size,
        size_t max_reserved_size)
        : 
        op_id(op_id),
        is_alloc(is_alloc),
        is_free(is_free),
        is_release(is_release),
        is_split(is_split),
        allocated_size(allocated_size),
        max_allocated_size(max_allocated_size),
        reserved_size(reserved_size),
        max_reserved_size(max_reserved_size),
        utilization_ratio(((float)allocated_size) / reserved_size),
        fragmentation(0){}
};


struct BlockInfo {
    size_t size;
    uint64_t address;
    bool allocated;

    BlockInfo() = default;

    BlockInfo(size_t size, uint64_t address, bool allocated)
                : size(size), address(address), allocated(allocated) {}
};

struct SegmentInfo {
    uint64_t op_id;
    uint64_t address;
    size_t total_size;
    Block* first_block;

    size_t allocated_size;
    size_t largest_freed_size;  // for fragmentation
    size_t num_blocks;
    size_t num_allocated_blocks;

    float fragmentation;

    std::vector<size_t> empty_range;


    SegmentInfo() = default;

    SegmentInfo(
        uint64_t op_id,
        uint64_t address,
        size_t total_size,
        Block* first_block)
        :
        op_id(op_id),
        address(address),
        total_size(total_size),
        first_block(first_block),
        allocated_size(0),
        largest_freed_size(0),
        num_blocks(0),
        num_allocated_blocks(0),
        fragmentation(0) { empty_range = std::vector<size_t>(); }

    SegmentInfo(const SegmentInfo& other)
        : op_id(other.op_id),
        address(other.address),
        total_size(other.total_size),
        first_block(other.first_block),
        allocated_size(other.allocated_size),
        largest_freed_size(other.largest_freed_size),
        num_blocks(other.num_blocks),
        num_allocated_blocks(other.num_allocated_blocks),
        fragmentation(other.fragmentation),
        empty_range(other.empty_range) {}


    bool operator<(const SegmentInfo& other) const {
        return this->op_id < other.op_id;
    }
};

struct AllocatorInfo {
    Status blocks;
    Status segments;
    Status allocated_bytes;
    Status reserved_bytes;

    AllocatorInfo() = default;

    AllocatorInfo(Status blocks, Status segments, Status allocated_bytes,
        Status reserved_bytes)
        : blocks(blocks), segments(segments), allocated_bytes(allocated_bytes),
        reserved_bytes(reserved_bytes) {}

    AllocatorInfo(const AllocatorInfo& other)
        : AllocatorInfo(other.blocks, other.segments, other.allocated_bytes,
        other.reserved_bytes) {}
};

typedef std::map<SegmentInfo, std::vector<BlockInfo>> SnapShot;

class allocatorProf {
private:
    uint64_t op_id = 0;

    AllocatorInfo allocator_info;

    SnapShot allocator_snapshot;
    
    std::map<MemoryRange, SegmentInfo> memory_segments;

    std::map<size_t ,SnapShot> allocator_snapshot_history;

    std::vector<OpInfo> op_list;

    std::vector<AllocatorInfo> allocator_info_history;

    std::map<uint64_t, bool> op_type_list;

private:

    void update_status(Status& stat, int64_t amount);

    void update_block_change(Block* block, const MemoryRange range, SegmentInfo& segment);

    MemoryRange locate_segment(Block* block);

    void dump_allocator_snapshot_history(std::string filename);

    void dump_op_type_list(std::string filename);

public:
    allocatorProf();

    ~allocatorProf();

    void update_segment_create(Block* block, size_t size);

    void update_segment_release(Block* block);

    void update_block_allocate(Block* block);

    void update_block_free(Block* block, size_t size);
};


namespace DumpDebugging{
// To determine which dump function to call
typedef enum DebuggingInfoType {
    BLOCK_MALLOC_OP_HISTORY = 0,
    BLOCK_FREE_OP_HISTORY = 1, // block free or segment release
    SEGMENT_OP_HISTORY = 2,
    ACTIVE_SEGMENT_LAYOUT = 3,
    BLOCK_POOLS_SNAPSHOT = 4,
    DUBUGGING_INFO_TYPE_LIST = 5
} DebuggingInfoType_t;

void enableDumppingDebugInfo();

// BLOCK_MALLOC_OP_HISTORY
// tuple: <real_alloc, is_split, orig_size, size, alloc_size, before_split_size, cur_allocated, cur_reserved>
typedef std::tuple<bool, bool, size_t, size_t, size_t, size_t, size_t, size_t> block_malloc_op_t;
void dump_block_malloc_op(bool is_simulator, const block_malloc_op_t& info);

// BLOCK_FREE_OP_HISTORY
// tuple: <is_release, bolck_size, cur_allocated, cur_reserved>
typedef std::tuple<bool, size_t, size_t, size_t> block_free_op_t;
void dump_block_free_op(bool is_simulator, const block_free_op_t& info);

// SEGMENT_OP_HISTORY
// tuple: <is_realese, alloc_size>
typedef std::tuple<bool, size_t> segment_op_t;
void dump_segment_op(bool is_simulator, const segment_op_t& info);

// ACTIVE_SEGMENT_LAYOUT
// tuple: <ptr, size, _active_segments>
typedef std::tuple<uint64_t, size_t, std::map<uint64_t, std::pair<op_id_t, size_t>>> segment_layout_t;
void dump_segment_layout(bool is_simulator, const segment_layout_t& info);

// BLOCK_POOLS_SNAPSHOT
// tuple: <small_pool, large_pool>
typedef std::tuple<BlockPoolSet_t, BlockPoolSet_t> block_pools_snapshot_t;
void dump_block_pools_snapshot(bool is_simulator, const block_pools_snapshot_t& info);


template<typename F>
void dumpDebuggingInfo(DebuggingInfoType_t type, F fn) {
    if (SimulatorModeController::is_debug_dumpping()) {
        if (type == BLOCK_MALLOC_OP_HISTORY) {
            fn();
        } else if (type == BLOCK_FREE_OP_HISTORY) {
            fn();
        } else if (type == SEGMENT_OP_HISTORY) {
            fn();
        } else if (type == ACTIVE_SEGMENT_LAYOUT) {
            fn();
        } else if (type == BLOCK_POOLS_SNAPSHOT) {
            fn();
        }
    }
}


}// namespace DumpDebugging

}  // namespace AllocatorSim
}  // namespace cuda
}  // namespace c10

#endif  // ALLOCATOR_PROFILER_H
