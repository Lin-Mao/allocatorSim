#include "allocator_profiler.h"
#include <filesystem>
#include <iostream>

namespace c10 {
namespace cuda {
namespace AllocatorSim {

allocatorProf::allocatorProf() {
}

allocatorProf::~allocatorProf() {
    ALLOCATOR_PROF_ENABLE();

    std::string path = "./output/";

    // in case the directory is not created
    if (!std::filesystem::is_directory(path)) {
        int ret = system(("mkdir -p " + path).c_str());
        if (ret != 0) {}
    }

    dump_allocator_snapshot_history(path + "snapshot_history.txt");

    dump_op_type_list(path + "op_type_list.log");
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

    op_type_list.emplace(op_id, true);

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

void allocatorProf::update_block_free(Block* block, size_t size) {
    ALLOCATOR_PROF_ENABLE();

    op_type_list.emplace(op_id, false);

    update_status(allocator_info.blocks, -1);
    update_status(allocator_info.allocated_bytes, size);
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
                //    << ", address: " << segment.first.address
                   << ", size: " << segment.first.total_size
                   << "B (" << format_size(segment.first.total_size) << ")"
                //    << ", ratio: "
                //    << ((float) segment.first.allocated_size) / segment.first.total_size
                //    << ", frag: " << segment.first.fragmentation
                   << std::endl;
            for (auto block : segment.second) {
                // output << "[" << block.address << ", "
                //        << block.address + block.size << "], "
                output << "size: " << block.size
                       << " B (" << format_size(block.size) << ")"
                       << ", used: " << std::boolalpha << block.allocated << std::endl;
            }
            output << std::endl;
        }
        output << std::endl;
    }
    output.close();
}

void allocatorProf::dump_op_type_list(std::string filename) {
    std::ofstream out(filename);
    for (auto op : op_type_list) {
        out << op.first << ": " << std::boolalpha << op.second << std::endl;
    }
    out.close();
}

namespace DumpDebugging {
namespace {
    std::string dump_path = "";

    std::string block_op_history_filename = "block_ops.txt";

    std::string segment_op_history_filename = "segment_ops.txt";

    std::string segment_layout_filename = "segment_layout.txt";

    std::string pools_snapshot_filename = "pools_snapshot.txt";

    std::vector<std::string> filename_list = {
        block_op_history_filename,
        segment_op_history_filename,
        segment_layout_filename,
        pools_snapshot_filename
    };

}   // anonymous namespace for variables

inline std::string get_filename_prefix(bool is_simulator) {
    std::string prefix = dump_path;
    if (is_simulator) {
        prefix += "simulator_";
    } else {
        prefix += "allocator_";
    }
    return prefix;
}

void flush_files() {
    for (auto filename : filename_list) {
        std::ofstream simulator_output(get_filename_prefix(true) + filename);
        simulator_output.close();
        std::ofstream allocator_output(get_filename_prefix(false) + filename);
        allocator_output.close();
    }
}

void enableDumppingDebugInfo() {
    dump_path = "./output/";

    // in case the directory is not created
    if (!std::filesystem::is_directory(dump_path)) {
        int ret = system(("mkdir -p " + dump_path).c_str());
        if (ret != 0) {}
    }

    flush_files();
}

void dump_block_malloc_op(bool is_simulator, const block_malloc_op_t& info) {
    std::ofstream output(get_filename_prefix(is_simulator) + block_op_history_filename, std::ios::app);
    output << "op_id: " << get_global_op_id() << std::boolalpha
           << ", alloc: " << std::get<0>(info)
           << ", split: " << std::get<1>(info)
           << ", orig_size: " << std::get<2>(info)
           << ", size: " << std::get<3>(info)
           << ", alloc_size: " << std::get<4>(info)
           << ", before_split_size: " << std::get<5>(info)
           << ", cur_allocated: " << std::get<6>(info)
           << ", cur_reserved: " << std::get<7>(info)
           << std::endl;
    output.close();
}

void dump_block_free_op(bool is_simulator, const block_free_op_t& info) {
    std::ofstream output(get_filename_prefix(is_simulator) + block_op_history_filename, std::ios::app);
    output << "op_id: " << get_global_op_id() << std::boolalpha
           << ", free: " << !std::get<0>(info)
           << ", release: " << std::get<0>(info)
           << ", size: " << std::get<1>(info)
           << ", cur_allocated: " << std::get<2>(info)
           << ", cur_reserved: " << std::get<3>(info)
           << std::endl;
    output.close();
}

void dump_segment_op(bool is_simulator, const segment_op_t& info) {
    std::string op_type;
    std::get<1>(info) ? op_type = "release" : op_type = "alloc";
    std::ofstream output(get_filename_prefix(is_simulator) + segment_op_history_filename, std::ios::app);
    output << "op_id: " << get_global_op_id() << ", " << op_type << ", alloc_size: " << std::get<1>(info) << std::endl;
    output.close();
}

void dump_segment_layout(bool is_simulator, const segment_layout_t& info) {
    std::ofstream output(get_filename_prefix(is_simulator) + segment_layout_filename, std::ios::app);
    output << "op_id: " << get_global_op_id() 
           << ", ptr: " << std::get<0>(info)
           << ", size: " << std::get<1>(info) << std::endl;
    for (auto i : std::get<2>(info)) {
        output << "[" << i.first << ", " << i.first + i.second.second << ") ";
    }
    output << std::endl;
    output.close();
}

void dump_block_pools_snapshot(bool is_simulator, const block_pools_snapshot_t& info) {
    std::ofstream output(get_filename_prefix(is_simulator) + pools_snapshot_filename, std::ios::app);
    output << "op_id: " << get_global_op_id() << std::endl;
    output << "small: ";
    for (auto b : std::get<0>(info)) {
        output << b->size << " ";
    }
    output << std::endl;

    output << "large: ";
    for (auto b : std::get<1>(info)) {
        output << b->size << " ";
    }
    output << std::endl;
    output.close();
}


}  // namespace dumpDebugging

}  // namespace AllocatorSim
}  // namespace cuda
}  // namespace c10
