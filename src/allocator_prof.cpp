#include "allocator_prof.h"

allocatorProf::allocatorProf() {
}

void allocatorProf::update_segment_allocate(Block* block, size_t size) {
    ALLOCATOR_PROF_ENABLE();
    // auto segment = std::make_shared<SegmentInfo>(op_id, nullptr, size, block);

}

void allocatorProf::update_segment_release(Block* block) {
    ALLOCATOR_PROF_ENABLE();

}

void allocatorProf::update_block_change() {
    ALLOCATOR_PROF_ENABLE();

}

void allocatorProf::update_block_allocate(Block* block) {
    ALLOCATOR_PROF_ENABLE();

}

void allocatorProf::update_block_free(Block* block) {
    ALLOCATOR_PROF_ENABLE();

}