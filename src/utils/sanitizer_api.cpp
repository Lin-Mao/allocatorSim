#include <allocator_utils.h>
#include <allocator_profiler.h>
#include <utils/sanitizer_api.h>
#include <sanitizer.h>
#include <iostream>
#include <fstream>
#include <functional>

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

            auto layout_info = std::make_tuple(data->address, data->size, _active_physical_segments);
            c10::cuda::AllocatorSim::DumpDebugging::dumpDebuggingInfo(
                c10::cuda::AllocatorSim::DumpDebugging::ACTIVE_SEGMENT_LAYOUT,
                std::bind(&c10::cuda::AllocatorSim::DumpDebugging::dump_segment_layout, false, layout_info)
            );

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

            auto layout_info = std::make_tuple(data->address, data->size, _active_physical_segments);
            c10::cuda::AllocatorSim::DumpDebugging::dumpDebuggingInfo(
                c10::cuda::AllocatorSim::DumpDebugging::ACTIVE_SEGMENT_LAYOUT,
                std::bind(&c10::cuda::AllocatorSim::DumpDebugging::dump_segment_layout, false, layout_info)
            );

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
