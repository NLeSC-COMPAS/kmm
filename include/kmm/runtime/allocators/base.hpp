#pragma once

#include <deque>
#include <memory>

#include "kmm/runtime/stream_manager.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {


enum struct AllocationResult {
    Success,
    ErrorOutOfMemory,
};

/**
 * Abstract base class for all asynchronous memory allocators.
 */
class AsyncAllocator {
  public:
    virtual ~AsyncAllocator() = default;

    /**
     *  Allocates `nbytes` of memory and returns the pointer in `addr_out`. The allocated
     *  region can only be used after the events in `deps_out` have completed.
     *
     *  Returns `AllocationResult::Success` if the operation was successful.
     */
    virtual AllocationResult allocate_async(
        size_t nbytes,
        void** addr_out,
        DeviceEventSet& deps_out
    ) = 0;

    /**
     * Deallocates the give address. This address MUST be previously allocated using
     * `allocated_async` with the exact same size. The `deps` parameter can be used to specify any
     * dependencies that must be satisfied e the memory is actually deallocated
     */
    virtual void deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps = {}) = 0;

    /**
     * Perform any pending asynchronous operations.
     */
    virtual void make_progress() {}

    /**
     * Trim unused memory to reduce the allocator's footprint.
     */
    virtual void trim(size_t nbytes_remaining = 0) {}
};
/**
 * Abstract base class for all synchronous memory allocators.
 */
class SyncAllocator: public AsyncAllocator {
    KMM_NOT_COPYABLE_OR_MOVABLE(SyncAllocator)

  public:
    SyncAllocator(
        std::shared_ptr<DeviceStreamManager> streams,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );

    ~SyncAllocator();

    /**
     * Allocate `nbytes` bytes of memory and sets the pointer in `addr_out`.
     *
     *  Returns `AllocationResult::Success` if the operation was successful.
     */
    virtual AllocationResult allocate(size_t nbytes, void** addr_out) = 0;

    /**
     * Deallocates the give address. This address MUST be previously allocated using
     * `allocate` with the exact same size.
     */
    virtual void deallocate(void* addr, size_t nbytes) = 0;

    AllocationResult allocate_async(size_t nbytes, void** addr_out, DeviceEventSet& deps_out) final;
    void deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) final;
    void make_progress() final;
    void trim(size_t nbytes_remaining = 0) final;

  private:
    struct DeferredDealloc {
        void* addr;
        size_t nbytes;
        DeviceEventSet dependencies;
    };

    std::shared_ptr<DeviceStreamManager> m_streams;
    std::deque<DeferredDealloc> m_pending_deallocs;
    size_t m_bytes_limit;
    size_t m_bytes_in_use = 0;
};

}  // namespace kmm