#ifndef ALLOCATOR_OPT_H
#define ALLOCATOR_OPT_H

#include "allocator_mgr.h"

class allocatorOpt {
private:
    allocatorMgr alloc_mgr;
    blockMap_t trace;
    uint64_t max;
    uint64_t min;
    size_t current_max_size;

    std::set<size_t> kMinBlockSize_candidates {256, 512, 1024, 2048, 4096};
    std::set<size_t> kSmallSize_candidates {1048576/2, 1048576, 1048576*2};
    std::set<size_t> kSmallBuffer_candidates {2097152, 2097152*2, 2097152*3, 2097152*4, 2097152*5};
    std::set<size_t> kLargeBuffer_candidates {20971520/2, 20971520, 20971520*3/2, 20971520*2, 20971520*5/2, 20971520*3};
    std::set<size_t> kMinLargeAlloc_candidates {1048576*2, 10485760*4, 10485760*6, 10485760*8, 10485760*10};
    std::set<size_t> kRoundLarge_candidates {2097152, 2097152*2, 2097152*4, 2097152*8, 2097152*16};

public:
    allocatorOpt(blockMap_t trace, uint64_t max, uint64_t min);

    void search_kMinBlockSize();
    
    void search_kSmallSize();

    void search_kSmallBuffer();

    void search_kLargeBuffer();

    void search_kMinLargeAlloc();

    void search_kRoundLarge();

    void search_configs();

    size_t evaluate_model();

    void report_config();

};

#endif  // ALLOCATOR_OPT_H