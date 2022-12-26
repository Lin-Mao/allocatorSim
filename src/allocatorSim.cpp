/**
 * @brief PyTorch CUDA caching allocator
 * @date 12/19/2022
*/

#include <array>
#include <cassert>

#include "allocatorSim.h"

allocatorSim::allocatorSim(){
    small_blocks = BlockPool(BlockComparator, true);
    large_blocks = BlockPool(BlockComparator, false);
}

allocatorSim::~allocatorSim() {
}

void allocatorSim::test_allocator() {
    std::cout << "Hello allocator!" << std::endl;
}

size_t allocatorSim::round_size(size_t size) {
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    } else if (size > allocator_config.roundup_bypass_threshold()) {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
    } else {
        auto divisions = allocator_config.roundup_power2_divisions();
        if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        // return roundup_power2_next_division(size, divisions);
        // not taken
        return size;
        } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        }
    }
}

BlockPool& allocatorSim::get_pool(size_t size, int stream) {
    if (size <= kSmallSize) {
        return small_blocks;
    } else {
        return large_blocks;
    }
}

size_t allocatorSim::get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
        return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
        return kLargeBuffer;
    } else {
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

bool allocatorSim::get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
        return false;
    if ((p.size() >= allocator_config.max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
        return false;
    if ((p.size() >= allocator_config.max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
        return false;
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;
}

bool allocatorSim::trigger_free_memory_callbacks(AllocParams& p) {
    // @todo(Lin-Mao): update reference?
    return true;
}

void allocatorSim::garbage_collect_cached_blocks() {
    // skip
}

bool allocatorSim::alloc_block(AllocParams& p, bool isRetry) {
    size_t size = p.alloc_size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, nullptr);
    return true;
}

bool allocatorSim::release_available_cached_blocks(AllocParams& p) {
    // @todo(Lin-Mao): todo
    return false;
}

bool allocatorSim::release_cached_blocks() {
    // @todo(Lin-Mao): todo
    return false;
}

bool allocatorSim::should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
        return remaining >= kMinBlockSize;
    } else {
        return (size < allocator_config.max_split_size()) &&
            (remaining > kSmallSize);
    }
}

Block* allocatorSim::malloc(int device, size_t orig_size, int stream) {
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size);

    bool block_found = get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params));

    if (!block_found) {
        // Do garbage collection if the flag is set.
        if (UNLIKELY(allocator_config.garbage_collection_threshold() > 0.0)) {
            garbage_collect_cached_blocks();
        }
        // Attempt allocate
        block_found = alloc_block(params, false)
            // Free enough available cached blocks to satisfy alloc and retry
            // alloc.
            || (release_available_cached_blocks(params) &&
                alloc_block(params, false))
            // Free all non-split cached blocks and retry alloc.
            || (release_cached_blocks() && alloc_block(params, true));
    }

    if (!block_found) {
        // OOM
        assert(block_found);
    }

    Block* block = params.block;
    Block* remaining = nullptr;

    // const bool already_split = block->is_split();

    if (should_split(block, size)) {
        remaining = block;
        block = new Block(device, stream, size, &pool, block->ptr);
        block->prev = remaining->prev;
        if (block->prev) {
        block->prev->next = block;
        }
        block->next = remaining;

        remaining->prev = block;
        // remaining->ptr = static_cast<char*>(remaining->ptr) + size;
        remaining->size -= size;
        bool inserted = pool.blocks.insert(remaining).second;
    }

    block->allocated = true;

    return block;
}

size_t allocatorSim::try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated) {
        return 0;
    }

    if (dst->prev == src) { // [src dst]
        // dst->ptr = src->ptr;
        dst->prev = src->prev;
        if (dst->prev) {
            dst->prev->next = dst;
        }
    } else { // [dest src]
        dst->next = src->next;
        if (dst->next) {
            dst->next->prev = dst;
        }
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    delete src;

    return subsumed_size;
}

void allocatorSim::free_block(Block* block) {
    size_t original_block_size = block->size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};

    for (Block* merge_candidate : merge_candidates) {
    const int64_t subsumed_size = try_merge_blocks(block, merge_candidate, pool);
    if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
    }
    }

    bool inserted = pool.blocks.insert(block).second;
}

void allocatorSim::free(Block* block) {
    block->allocated = false;

    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    free_block(block);
}