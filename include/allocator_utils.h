/**
 * Utilities (struct, datatype, common function definition) of allocator.
*/
#ifndef ALLOCATOR_UTILS_H
#define ALLOCATOR_UTILS_H

#include <cstddef>
#include <cstdint>
#include <set>
#include <map>
#include <vector>
#include <array>
#include <memory>
#include <sstream>

#define LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))

struct Block;
struct BlockPool;
struct AllocParams;
struct Status;
struct OpInfo;
struct BlockInfo;
struct SegmentInfo;

typedef bool (*Comparison)(const Block*, const Block*);

struct Block {
    int device; // gpu
    int stream; // allocation stream
    std::set<size_t> stream_uses; // streams on which the block was used
    size_t size; // block size in bytes
    BlockPool* pool; // owning memory pool
    uint64_t ptr; // memory address
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
        uint64_t ptr)
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
            ptr(0),
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

struct Status {
    int64_t current = 0;
    int64_t peak = 0;
    int64_t allocated = 0;
    int64_t freed = 0;
};

enum struct StatusType : size_t {
    AGGREGATE = 0,
    SMALL_POOL = 1,
    LARGE_POOL = 2,
    NUM_TYPES = 3
};

typedef std::array<Status, static_cast<size_t>(StatusType::NUM_TYPES)> StatusAarry;

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
  size_t address;
  bool allocated;

  BlockInfo() = default;

  BlockInfo(size_t size, size_t address, bool allocated)
            : size(size), address(address), allocated(allocated) {}
};

struct SegmentInfo {
  uint64_t op_id;
  size_t address;
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
    size_t address,
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

    bool operator<(const SegmentInfo& other) const {
        return this->op_id < other.op_id;
    }
};

struct MemoryRange {
  size_t start;
  size_t end;

  MemoryRange() = default;

  MemoryRange(size_t start, size_t end) : start(start), end(end) {}

  bool operator<(const MemoryRange &other) const {
    return this->start < other.start;
  }
};

std::string format_size(size_t size);

#endif // ALLOCATOR_UTILS_H