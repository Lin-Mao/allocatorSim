#ifndef ALLOCATOR_CONFIG_H
#define ALLOCATOR_CONFIG_H

#include <cstddef>
#include <atomic>
#include <limits>

// @todo(Lin-Mao) make all the field static (shared by different allocator instance)
struct allocatorConfig {
 public:
  size_t max_split_size() {
    return this->m_max_split_size;
  }
  size_t garbage_collection_threshold() {
    return this->m_garbage_collection_threshold;
  }

  size_t roundup_power2_divisions() {
    return this->m_roundup_power2_divisions;
  }
  size_t roundup_bypass_threshold() {
    return this->m_roundup_bypass_threshold;
  }

  allocatorConfig() {
    m_max_split_size = std::numeric_limits<size_t>::max();
    m_roundup_power2_divisions = 0;
    m_roundup_bypass_threshold = std::numeric_limits<size_t>::max();
    m_garbage_collection_threshold = 0;
  }

 private:
  std::atomic<size_t> m_max_split_size;
  std::atomic<size_t> m_roundup_power2_divisions;
  std::atomic<size_t> m_roundup_bypass_threshold;
  std::atomic<double> m_garbage_collection_threshold;
};

#endif // ALLOCATOR_CONFIG_H