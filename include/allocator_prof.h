/**
 * The states of allocator.
*/
#ifndef ALLOCATOR_PROF_H
#define ALLOCATOR_PROF_H

#define DISABLE_ALLOCATOR_PROF 0

#if DISABLE_ALLOCATOR_PROF
#define ALLOCATOR_PROF_ENABLE() return
#else
#define ALLOCATOR_PROF_ENABLE() ((void)0)
#endif

#include "allocator_utils.h"

#include <fstream>

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

private:

    void update_status(Status& stat, int64_t amount);

    void update_block_change(Block* block, const MemoryRange range, SegmentInfo& segment);

    MemoryRange locate_segment(Block* block);

    void dump_allocator_snapshot_history(std::string filename);

public:
    allocatorProf();

    ~allocatorProf();

    void update_segment_create(Block* block, size_t size);

    void update_segment_release(Block* block);

    void update_block_allocate(Block* block);

    void update_block_free(Block* block);
};


#endif  // ALLOCATOR_PROF_H