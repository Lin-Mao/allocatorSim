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
#include <chrono>
#include <unordered_map>
#include <tuple>

// GCC version should be greater than 5.0
// handle filesystem on old compilers
#if __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#endif

namespace c10 {
namespace cuda {
namespace AllocatorSim {

/******************************************************************************/
/******************************* Common Macros ********************************/
/******************************************************************************/
#define LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))


/******************************************************************************/
/******************************* Common DataType ******************************/
/******************************************************************************/
typedef uint64_t op_id_t;

// <malloc_op_id, <free_op_id, size>>
typedef std::map<op_id_t, std::pair<op_id_t, size_t>> trace_t;

struct Block;
struct BlockPool;

typedef bool (*Comparison)(const Block*, const Block*);

typedef std::set<Block*, Comparison> BlockPoolSet_t;
struct Block {
    int device; // gpu
    int stream; // allocation stream
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
            size(size),
            pool(nullptr),
            ptr(0),
            allocated(0),
            prev(nullptr),
            next(nullptr),
            event_count(0),
            gc_count(0) {}

    // constructor for dump_block_pools_snapshot
    Block(int device, int stream, size_t size, uint64_t ptr)
        : device(device),
            stream(stream),
            size(size),
            pool(nullptr),
            ptr(ptr),
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


struct MemoryRange {
    size_t start;
    size_t end;

    MemoryRange() = default;

    MemoryRange(size_t start, size_t end) : start(start), end(end) {}

    bool operator<(const MemoryRange &other) const {
        return this->start < other.start;
    }
};


/******************************************************************************/
/****************************** Common Functions ******************************/
/******************************************************************************/
// format size to human readable string
std::string format_size(size_t size);

// get and increase global op_id
op_id_t get_global_op_id();
void increase_global_op_id();


/******************************************************************************/
/****************************** Allocator Timer *******************************/
/******************************************************************************/
#define TIMER_NUMS 10
using sys_clock = std::chrono::time_point<std::chrono::system_clock>;

struct allocatorTimer {
private:
    static std::array<size_t, TIMER_NUMS> timers;
    static std::array<std::string, TIMER_NUMS> timer_names;
    static std::array<sys_clock, TIMER_NUMS> starts;
    static std::array<sys_clock, TIMER_NUMS> ends;

public:
    static void start_timer(int index);
    static void stop_timer(int index);
    static void log_timer(int index, std::string name);
    static void print_timer(int index);
};


/******************************************************************************/
/************************** SimulatorModeController ***************************/
/******************************************************************************/
namespace sim_control{
typedef enum SimControlMode{
    ASYNC_TRACING = 0,
    FUNCTIONALITY_CHECKING = 1,
    PROFILING = 2,
    STATIC_TENSOR_ANALYSIS = 3,
    DEBUG_DUMPPING = 4,
    DEBUG_POOLINFO_DUMPPING = 5,
    TRACE_DUMPPING = 6,
    CONFIG_OPTIMIZATION = 7,
    GROUP_OPTIMIZATION = 8,
    NUMS_OF_SIM_CONTROL_MODE = 9
}SimControlMode_t;

void set_sim_control_mode(SimControlMode_t mode, bool enable);

// the following struct is used to control the simulator mode
struct SimulatorModeController{
    /*
    init the an arrangement of control flags
    */
   static void init();

   static void show();

    /*
    control the way to collect trace
    true: async, optimization and tracing is in this mode
    false: sync (only for correctness checking)
    */
    static bool enable_async_tracing;
    static bool is_async_tracing();
    static void set_async_tracing(bool async);

    /*
    If true, the simulator will check the correctness of the functionality,
    and do not apply any optimization.
    */
    static bool enable_functionality_checking;
    static bool is_functionality_checking();
    static void set_functionality_checking(bool checking);

    /*
    It's controlled by torch.cuda.enable_profiling() API
    true, collecting trace and searching configuration
    false, applying optimization 
    */
    static bool enable_profiling;
    static bool is_profiling();
    static void set_profiling(bool profiling);

    /*
    deploy the static tensor analysis or not
    true, anaylzing static tensor
    false, not analyzing static tensor
    */
    static bool enable_static_tensor_analysis;
    static bool is_static_tensor_analysis();
    static void set_static_tensor_analysis(bool analysis);

    /*
    control dumping trace to file for simulation and allocator
    */
    static bool enable_debug_dumpping;
    static bool is_debug_dumpping();
    static void set_debug_dumpping(bool dumpping);

    /*
    Separate dumping pool snapshot because of reducing overhead
    */
    static bool enable_debug_poolinfo_dumpping;
    static bool is_debug_poolinfo_dumpping();
    static void set_debug_poolinfo_dumpping(bool dumpping);

    /*
    control dumping trace to file
    Only enable when dumpping trace to file for exploring reinforcement learning
    */
    static bool enable_trace_dumpping;
    static bool is_trace_dumpping();
    static void set_trace_dumpping(bool dumpping);

    /*
    enable config searching
    */
    static bool enable_config_optimization;
    static bool is_config_optimization();
    static void set_config_optimization(bool optimization);

    /*
    enable group optimization
    */
    static bool enable_group_optimization;
    static bool is_group_optimization();
    static void set_group_optimization(bool optimization);
};

}  // namespace sim_control

}  // namespace AllocatorSim
}  // namespace cuda
}  // namespace c10

#endif // ALLOCATOR_UTILS_H