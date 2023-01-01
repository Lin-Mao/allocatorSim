#ifndef ALLOCATOR_MGR_H
#define ALLOCATOR_MGR_H

#include "allocator_utils.h"
#include "allocator_sim.h"

class allocatorMgr {
private:
    int device;
    int stream;
    allocatorSim alloc_sim;
    std::map<Block*, size_t> block_ref_map;
    
public:
    allocatorMgr();

    allocatorMgr(int device, int stream);

    void malloc_block(size_t orig_size, size_t ref);

    void update_block_reference();

    void free_block();

    void empty_cache();

    std::pair<size_t, size_t> get_allocator_memory_usage();

    void reset_allocator_memory_usage();

    void show_allocator_memory_usage();

};


#endif  // ALLOCATOR_MGR_H