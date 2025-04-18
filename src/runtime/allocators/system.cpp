#include "kmm/runtime/allocators/system.hpp"

namespace kmm {

AllocationResult SystemAllocator::allocate(size_t nbytes, void** addr_out) {
    *addr_out = malloc(nbytes);
    return *addr_out != nullptr ? AllocationResult::Success : AllocationResult::ErrorOutOfMemory;
}

void SystemAllocator::deallocate(void* addr, size_t nbytes) {
    free(addr);
}
}  // namespace kmm