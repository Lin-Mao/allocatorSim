#ifndef SANITIZER_API_H
#define SANITIZER_API_H

#include <map>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <string>

extern size_t global_op_id;

// <address, <op_id, size>>
extern std::map<uint64_t, std::pair<uint64_t, size_t>> _active_physical_segments;

// <malloc_op_id, <free_op_id, size>>
extern std::map<uint64_t, std::pair<uint64_t, size_t>> _physical_segment_trace;

void sanitizer_callbacks_subscribe();

void sanitizer_callbacks_unsubscribe();

#endif  // SANITIZER_API_H