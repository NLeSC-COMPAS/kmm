#pragma once

#include <memory>
#include <unordered_map>

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class CachingAllocator: public AsyncAllocator {
  public:
    CachingAllocator(std::unique_ptr<AsyncAllocator> allocator, double max_fragmentation=0.75, size_t initial_watermark = 0);
    ~CachingAllocator();
    AllocationResult allocate_async(size_t nbytes, void** addr_out, DeviceEventSet& deps_out) final;
    void deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) final;
    void make_progress() final;
    void trim(size_t nbytes_remaining = 0) final;
    size_t free_some_memory();

  private:
    bool can_allocate_bytes(size_t nbytes) const;

    struct AllocationSlot;
    struct AllocationBin {
        std::unique_ptr<AllocationSlot> head;
        AllocationSlot* tail;
    };

    std::unique_ptr<AsyncAllocator> m_allocator;
    std::unordered_map<size_t, AllocationBin> m_allocation_bins;
    AllocationSlot* m_lru_oldest = nullptr;
    AllocationSlot* m_lru_newest = nullptr;
    size_t m_bytes_watermark = 0;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_allocated = 0;
    double m_max_fragmentation = 0.75;
};

}  // namespace kmm