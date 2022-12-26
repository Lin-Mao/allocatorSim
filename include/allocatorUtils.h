#ifndef ALLOCATOR_UTILS_H
#define ALLOCATOR_UTILS_H

#include <cstddef>
#include <cstdint>
#include <set>
#include <memory>

#define LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

// all sizes are rounded to at least 512 bytes
constexpr size_t kMinBlockSize =512;
// largest "small" allocation is 1 MiB
constexpr size_t kSmallSize = 1048576;
// "small" allocations are packed in 2 MiB blocks
constexpr size_t kSmallBuffer = 2097152;
// "large" allocations may be packed in 20 MiB blocks
constexpr size_t kLargeBuffer = 20971520;
// allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kMinLargeAlloc = 10485760;
// round up large allocations to 2 MiB
constexpr size_t kRoundLarge = 2097152;

struct Block;
struct BlockPool;
struct AllocParams;

typedef bool (*Comparison)(const Block*, const Block*);

struct Block {
    int device; // gpu
    int stream; // allocation stream
    std::set<size_t> stream_uses; // streams on which the block was used
    size_t size; // block size in bytes
    BlockPool* pool; // owning memory pool
    void* ptr; // memory address
    bool allocated; // in-use flag
    Block* prev; // prev block if split from a larger allocation
    Block* next; // next block if split from a larger allocation
    int event_count; // number of outstanding CUDA events
    int gc_count; // counter for prioritizing older / less useful blocks for
                    // garbage collection

    Block(
        int device,
        int stream,
        size_t size,
        BlockPool* pool,
        void* ptr)
        : device(device),
            stream(stream),
            stream_uses(),
            size(size),
            pool(pool),
            ptr(ptr),
            allocated(0),
            prev(nullptr),
            next(nullptr),
            event_count(0),
            gc_count(0) {}

    // constructor for search key
    Block(int device, int stream, size_t size)
        : device(device),
            stream(stream),
            stream_uses(),
            size(size),
            pool(nullptr),
            ptr(nullptr),
            allocated(0),
            prev(nullptr),
            next(nullptr),
            event_count(0),
            gc_count(0) {}

    bool is_split() const {
        return (prev != nullptr) || (next != nullptr);
    }
};

struct BlockPool {
    std::set<Block*, Comparison> blocks;
    bool is_small;

    BlockPool() = default;

    BlockPool(
            Comparison comparator,
            bool small)
            : blocks(comparator), is_small(small) {}
};

struct AllocParams {
  AllocParams(
        int device,
        size_t size,
        int stream,
        BlockPool* pool,
        size_t alloc_size)
        : search_key(device, stream, size),
            pool(pool),
            alloc_size(alloc_size),
            block(nullptr) {}

    int device() const {
        return search_key.device;
    }
    int stream() const {
        return search_key.stream;
    }
    size_t size() const {
        return search_key.size;
    }

    Block search_key;
    BlockPool* pool;
    size_t alloc_size;
    Block* block;
};

static bool BlockComparator(const Block* a, const Block* b) {
    if (a->stream != b->stream) {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
    }
    if (a->size != b->size) {
        return a->size < b->size;
    }
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

static bool BlockComparator(const Block* a, const Block* b);

#endif // ALLOCATOR_UTILS_H