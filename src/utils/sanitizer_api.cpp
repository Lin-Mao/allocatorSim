#include <allocator_utils.h>
#include <utils/sanitizer_api.h>
#include <sanitizer.h>
#include <iostream>
#include <fstream>

static Sanitizer_SubscriberHandle sanitizer_handle;

std::map<uint64_t, std::pair<uint64_t, size_t>> _active_physical_segments = {};

std::map<uint64_t, std::pair<uint64_t, size_t>> _physical_segment_trace = {};

bool skip_the_first_malloc_of_sanitizer = true;

void memory_callbacks(
    void* userdata,
    Sanitizer_CallbackDomain domain,
    Sanitizer_CallbackId cbid,
    const void* cbdata)
{
    if (domain != SANITIZER_CB_DOMAIN_RESOURCE) {
        return;
    }

    switch(cbid) {
        case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC:
        {
            // std::cout << "SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC" << std::endl;
            Sanitizer_ResourceMemoryData *data = (Sanitizer_ResourceMemoryData*)cbdata;
            if (skip_the_first_malloc_of_sanitizer && data->size == 512) {
                skip_the_first_malloc_of_sanitizer = false;
                break;
            }
            _active_physical_segments.emplace(data->address, std::make_pair(c10::cuda::AllocatorSim::get_global_op_id(), data->size));
            // std::cout << "Allocated " << data->size << " bytes at " << data->address << std::endl;

#ifdef DUMP_INFO_TO_FILE_DEBUGGING
            std::ofstream out1(c10::cuda::AllocatorSim::get_dump_file_path() + "allocator_mem_layout.txt", std::ios::app);
            out1 << "op_id: " << c10::cuda::AllocatorSim::get_global_op_id() << " malloc: " << data->address << " size: " << data->size << std::endl;
            for (auto s : _active_physical_segments) {
                out1 << "[" << s.first << ", " << s.first + s.second.second << ") ";
            }
            out1 << std::endl;
#endif
            break;
        }
        case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE:
        {
            // std::cout << "SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE" << std::endl;
            Sanitizer_ResourceMemoryData *data = (Sanitizer_ResourceMemoryData*)cbdata;
            auto s = _active_physical_segments.find(data->address);
            _physical_segment_trace.emplace(s->second.first, std::make_pair(c10::cuda::AllocatorSim::get_global_op_id(), s->second.second));
            // std::cout << "Freed " << s->second.second << " bytes at " << s->first << std::endl;
            _active_physical_segments.erase(s);

#ifdef DUMP_INFO_TO_FILE_DEBUGGING
            std::ofstream out(c10::cuda::AllocatorSim::get_dump_file_path() + "allocator_mem_layout.txt", std::ios::app);
            out << "op_id: " << c10::cuda::AllocatorSim::get_global_op_id() << " free: " << data->address << " size: " << data->size << std::endl;
            for (auto s : _active_physical_segments) {
                out << "[" << s.first << ", " << s.first + s.second.second << ") ";
            }
            out << std::endl;
#endif
            break;
        }
        case SANITIZER_CBID_RESOURCE_HOST_MEMORY_ALLOC:
        {
            std::cout << "SANITIZER_CBID_RESOURCE_HOST_MEMORY_ALLOC" << std::endl;
            break;
        }
        case SANITIZER_CBID_RESOURCE_HOST_MEMORY_FREE:
        {
            std::cout << "SANITIZER_CBID_RESOURCE_HOST_MEMORY_FREE" << std::endl;
            break;
        }
        case SANITIZER_CBID_RESOURCE_MEMORY_ALLOC_ASYNC:
        {
            std::cout << "SANITIZER_CBID_RESOURCE_MEMORY_ALLOC_ASYNC" << std::endl;
            break;
        }
        case SANITIZER_CBID_RESOURCE_MEMORY_FREE_ASYNC:
        {
            std::cout << "SANITIZER_CBID_RESOURCE_MEMORY_FREE_ASYNC" << std::endl;
            break;
        }
        case SANITIZER_CBID_RESOURCE_MEMORY_FREE_ASYNC_DONE:
        {
            std::cout << "SANITIZER_CBID_RESOURCE_MEMORY_FREE_ASYNC_DONE" << std::endl;
            break;
        }
        
    }
}

void sanitizer_callbacks_subscribe() {
    sanitizerSubscribe(&sanitizer_handle, memory_callbacks, nullptr);
    sanitizerEnableDomain(1, sanitizer_handle, SANITIZER_CB_DOMAIN_RESOURCE);
}

void sanitizer_callbacks_unsubscribe() {
    sanitizerUnsubscribe(sanitizer_handle);
    sanitizerEnableDomain(1, sanitizer_handle, SANITIZER_CB_DOMAIN_RESOURCE);
}
