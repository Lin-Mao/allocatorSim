#include <utils/sanitizer_api.h>
#include <sanitizer.h>
#include <iostream>

static Sanitizer_SubscriberHandle sanitizer_handle;

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
            std::cout << "SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC" << std::endl;
            break;
        }
        case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE:
        {
            std::cout << "SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE" << std::endl;
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
