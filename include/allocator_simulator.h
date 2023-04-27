/**
 * @brief PyTorch CUDA caching allocator simulator
 * @date 12/19/2022
*/
#ifndef ALLOCATOR_SIMULATOR_H
#define ALLOCATOR_SIMULATOR_H

#include <iostream>

#include "allocator_utils.h"
#include "allocator_config.h"
#include "allocator_profiler.h"

namespace c10 {
namespace cuda {
namespace AllocatorSim {

class allocatorSim {
private:
    BlockPool small_blocks;
    BlockPool large_blocks;
    uint64_t segment_address = allocatorConf::get_memory_segment_address_start();
    size_t max_reserved_bytes;
    size_t current_reserved_bytes;
    size_t max_allocated_bytes;
    size_t current_allocated_bytes;

    std::unordered_map<uint64_t, Block*> releasable_blocks;

    allocatorProf* allocator_prof;

    bool group_enable_flag_sim = false;

private:
    size_t round_size(size_t ori_size);

    BlockPool& get_pool(size_t size, int stream);
    
    size_t get_allocation_size(size_t size);

    bool get_free_block(AllocParams& p);

    bool trigger_free_memory_callbacks(AllocParams& p);

    void garbage_collect_cached_blocks();

    bool alloc_block(AllocParams& p, bool isRetry, void* o_ptr);

    bool release_available_cached_blocks(AllocParams& p);

    bool should_split(const Block* block, size_t size);

    void free_block(Block* block);

    size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool);

    void release_blocks(BlockPool& pool);

    size_t get_grouped_allocation_size_sim(size_t size);

public:
    allocatorSim();

    ~allocatorSim();

    void test_allocator();

    Block* malloc(int device, size_t orig_size, int stream, void* ptr = nullptr);

    void free(Block* block);

    bool release_cached_blocks();

    void release_block(Block* block);

    Block* retrieve_released_block(uint64_t ptr);

    std::pair<size_t, size_t> get_max_memory_usage();

    size_t get_max_reserved_bytes();

    size_t get_max_allocated_bytes();

    void set_max_memory_usage(size_t allocated_bytes, size_t reserved_bytes);

    void set_group_enable_flag_sim(bool flag);

};

}  // namespace c10
}  // namespace cuda
}  // namespace AllocatorSim

#endif  // ALLOCATOR_SIMULATOR_H
