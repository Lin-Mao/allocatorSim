#include "allocator_prof.h"

allocatorProf::allocatorProf() {
}

allocatorProf::~allocatorProf() {
    ALLOCATOR_PROF_ENABLE();

    dump_allocator_snapshot_history("/home/lm/allocatorSim/output/snapshot_history.txt");
}

void allocatorProf::update_segment_create(Block* block, size_t size) {
    ALLOCATOR_PROF_ENABLE();

    update_status(allocator_info.segments, 1);
    update_status(allocator_info.reserved_bytes, size);

    auto segment = SegmentInfo(op_id, block->ptr, size, block);
    memory_segments.emplace(MemoryRange(block->ptr, block->ptr + size), segment);
    allocator_snapshot.emplace(segment, std::vector<BlockInfo>());
}

void allocatorProf::update_segment_release(Block* block) {
    ALLOCATOR_PROF_ENABLE();

    update_status(allocator_info.segments, -1);
    update_status(allocator_info.reserved_bytes, block->size);

    auto range = locate_segment(block);
    auto segment = memory_segments.at(range);

    allocator_snapshot.erase(segment);
    memory_segments.erase(range);

    allocator_snapshot_history.emplace(op_id, SnapShot(allocator_snapshot));

}

void allocatorProf::update_block_change(Block* block, 
                                        const MemoryRange range,
                                        SegmentInfo& segment) {
    ALLOCATOR_PROF_ENABLE();

    auto& blocks = allocator_snapshot.at(segment);

    if (block->ptr == segment.address) {
        segment.first_block = block;
    }

    segment.num_blocks = 0;
    segment.num_allocated_blocks = 0;
    segment.allocated_size = 0;
    segment.largest_freed_size = 0;
    blocks.clear();

    Block* fblock = segment.first_block;
    while (fblock != nullptr) {
        blocks.push_back(
            BlockInfo(fblock->size, fblock->ptr, fblock->allocated)
        );

        if (fblock->allocated) {
            segment.allocated_size += fblock->size;
            segment.num_allocated_blocks++;
        } else {
            segment.largest_freed_size = 
                std::max(segment.largest_freed_size, fblock->size);
        }
        segment.num_blocks++;

        fblock = fblock->next;
    }
    if (segment.total_size - segment.allocated_size != 0) {
        segment.fragmentation = 1 - ((float)segment.largest_freed_size) 
                            / (segment.total_size - segment.allocated_size);
    } else {
        segment.fragmentation = 0;
    }

    allocator_snapshot_history.emplace(op_id, SnapShot(allocator_snapshot));

}

void allocatorProf::update_block_allocate(Block* block) {
    ALLOCATOR_PROF_ENABLE();

    update_status(allocator_info.blocks, 1);
    update_status(allocator_info.allocated_bytes, block->size);
    auto range = locate_segment(block);
    auto& segment = memory_segments.at(range);

    // @todo(Lin-Mao): can be eliminated?
    if (segment.address == block->ptr && !segment.empty_range.empty()) {
        segment.empty_range.push_back(op_id);
    }

    update_block_change(block, range, segment);

    allocator_info_history.push_back(AllocatorInfo(allocator_info));

    op_id++;
}

void allocatorProf::update_block_free(Block* block) {
    ALLOCATOR_PROF_ENABLE();

    update_status(allocator_info.blocks, -1);
    update_status(allocator_info.allocated_bytes, block->size);
    auto range = locate_segment(block);
    auto& segment = memory_segments.at(range);

    if (block->prev == nullptr || block->next == nullptr) {
        segment.empty_range.push_back(op_id);
    }

    update_block_change(block, range, segment);

    allocator_info_history.push_back(AllocatorInfo(allocator_info));

    op_id++;
}

void allocatorProf::update_status(Status& stat, int64_t amount) {
    stat.current += amount;

    stat.peak = std::max(stat.current, stat.peak);

    if (amount > 0) {
        stat.allocated += amount;
    } else {  // amount < 0
        stat.freed += -amount;
    }
}

MemoryRange allocatorProf::locate_segment(Block* block) {
    size_t len = memory_segments.size();
    MemoryRange address_map[len];
    // @todo(Lin-Mao): change to sorted set

    size_t count = 0;
    for (auto i : memory_segments) {
    address_map[count++] = i.first;
    }

    // binary search
    size_t low = 0, mid = 0, high = len - 1;
    while (low < high) {
        mid = (low + high + 1) >> 1; // Round up, otherwise infinite loop
        if (address_map[mid].start > (size_t) block->ptr) {
            high = (mid > 0) ? (mid - 1) : 0; // otherwise overflow
        } else {
            low = mid;
        }
    }

    if (address_map[low].start <= (size_t) block->ptr) {
        return address_map[low];
    } else {
    printf("It shouldn't take this branch. Please check!!!\n");
    return MemoryRange();
    }
}

void allocatorProf::dump_allocator_snapshot_history(std::string filename) {
    std::ofstream output;
    std::string pad(80, '#');

    output.open(filename);
    for (auto snapshot : allocator_snapshot_history) {
        output << "op_id: " << snapshot.first << " "
               << pad.c_str() << std::endl;
        for (auto segment : snapshot.second) {
            output << "[segment] "
                   << "op_id: " << segment.first.op_id
                   << ", address: " << segment.first.address
                   << ", size: " << segment.first.total_size
                   << "B (" << format_size(segment.first.total_size) << ")"
                   << ", ratio: "
                   << ((float) segment.first.allocated_size) / segment.first.total_size
                   << ", frag: " << segment.first.fragmentation
                   << std::endl;
            for (auto block : segment.second) {
                output << "[" << block.address << ", "
                       << block.address + block.size << "]"
                       << ", size: " << block.size
                       << " B (" << format_size(block.size) << ")"
                       << ", used: " << std::boolalpha << block.allocated << std::endl;
            }
            output << std::endl;
        }
        output << std::endl;
    }
    output.close();
}