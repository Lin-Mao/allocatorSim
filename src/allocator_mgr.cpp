#include "allocator_mgr.h"

allocatorMgr::allocatorMgr() : allocatorMgr(0, 0) {
}

allocatorMgr::allocatorMgr(int device, int stream) {
    this->device = device;
    this->stream = stream;
    this->block_ref_map = std::map<Block*, size_t>();
    allocatorSim alloc_sim();
}

void allocatorMgr::malloc_block(size_t orig_size, size_t ref) {
    Block* b = this->alloc_sim.malloc(this->device, orig_size, this->stream);
    this->block_ref_map.emplace(b, ref);
}

void allocatorMgr::update_block_reference() {
    for (auto& b : this->block_ref_map) {
        --b.second;
    }
}

void allocatorMgr::free_block() {
    std::vector<Block*> del_blocks;
    for (const auto b : this->block_ref_map) {
        if (b.second == 0) {
            this->alloc_sim.free(b.first);
            del_blocks.push_back(b.first);
        }
    }

    for (const auto b : del_blocks) {
        this->block_ref_map.erase(b);
    }
}

size_t allocatorMgr::get_reserved_size() {
    return alloc_sim.get_max_reserved_size();
}

void allocatorMgr::reset_reserved_size() {
    alloc_sim.reset_max_reserved_size();
}