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

struct AllocatorStatus {
    StatusAarry blocks;
    StatusAarry segments;
    StatusAarry allocated_bytes;
    StatusAarry reserved_bytes;
};

class allocatorProf {
private:
    uint64_t op_id = 0;

    AllocatorStatus allocator_status;
    
    std::map<MemoryRange, std::shared_ptr<SegmentInfo>> memory_segments;

public:
    allocatorProf();

    void update_segment_allocate(Block* block, size_t size);

    void update_segment_release(Block* block);

    void update_block_change();

    void update_block_allocate(Block* block);

    void update_block_free(Block* block);
};


#endif  // ALLOCATOR_PROF_H