#include "kmm/runtime/allocators/caching.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

CachingAllocator::CachingAllocator(
    std::unique_ptr<AsyncAllocator> allocator,
    double max_fragmentation,
    size_t initial_watermark
) :
    m_allocator(std::move(allocator)),
    m_bytes_watermark(initial_watermark),
    m_max_fragmentation(max_fragmentation){
    KMM_ASSERT(m_allocator != nullptr);
    KMM_ASSERT(m_max_fragmentation >= 0.0 && m_max_fragmentation < 1.0);
}

CachingAllocator::~CachingAllocator() {
    while (free_some_memory() > 0) {
        //
    }
}

struct CachingAllocator::AllocationSlot {
    AllocationSlot(void* addr, size_t nbytes, DeviceEventSet dependencies) :
        addr(addr),
        nbytes(nbytes),
        dependencies(std::move(dependencies)) {}

    void* addr = nullptr;
    size_t nbytes = 0;
    DeviceEventSet dependencies;
    std::unique_ptr<AllocationSlot> next = nullptr;
    AllocationSlot* lru_older = nullptr;
    AllocationSlot* lru_newer = nullptr;
};

size_t round_up_allocation_size(size_t nbytes) {
    if (nbytes >= 1024) {
        return round_up_to_multiple(nbytes, size_t(1024));
    } else {
        return round_up_to_power_of_two(nbytes);
    }
}

bool CachingAllocator::can_allocate_bytes(size_t nbytes) const {
    // If `m_bytes_allocated + nbytes <= m_bytes_watermark` then allocating nbytes will not
    // raise the watermark, so it is allowed
    if (nbytes <= m_bytes_watermark - m_bytes_allocated) {
        return true;
    }


    // If `m_bytes_allocated - m_bytes_in_use == 0` then there is no overhead at all, so we
    // must allow the allocation.
    auto overhead = m_bytes_allocated - m_bytes_in_use;
    if (overhead == 0) {
        return true;
    }

    // Otherwise, measure if the new overhead exceeds the maximum allowed fragmentation
    auto new_watermark = m_bytes_allocated + nbytes;
    return double(overhead) <= m_max_fragmentation * double(new_watermark);
}

AllocationResult CachingAllocator::allocate_async(
    size_t nbytes,
    void** addr_out,
    DeviceEventSet& deps_out
) {
    nbytes = round_up_allocation_size(nbytes);
    auto& bin = m_allocation_bins[nbytes];

    if (bin.head == nullptr) {
        while (true) {
            if (can_allocate_bytes(nbytes)) {
                auto result = m_allocator->allocate_async(nbytes, addr_out, deps_out);

                if (result == AllocationResult::Success) {
                    m_bytes_allocated += nbytes;
                    m_bytes_in_use += nbytes;
                    m_bytes_watermark = std::max(m_bytes_watermark, m_bytes_allocated);
                    return AllocationResult::Success;
                }
            }

            if (free_some_memory() == 0) {
                return AllocationResult::ErrorOutOfMemory;
            }
        }
    }

    auto slot = std::move(bin.head);

    if (slot->next != nullptr) {
        bin.head = std::move(slot->next);
    } else {
        bin.tail = nullptr;
    }

    if (slot->lru_older != nullptr) {
        slot->lru_older->lru_newer = slot->lru_newer;
    } else {
        m_lru_oldest = slot->lru_newer;
    }

    if (slot->lru_newer != nullptr) {
        slot->lru_newer->lru_older = slot->lru_older;
    } else {
        m_lru_newest = slot->lru_older;
    }

    m_bytes_in_use += nbytes;
    *addr_out = slot->addr;
    deps_out.insert(std::move(slot->dependencies));
    return AllocationResult::Success;
}

void CachingAllocator::deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) {
    nbytes = round_up_allocation_size(nbytes);
    m_bytes_in_use -= nbytes;

    auto slot = std::make_unique<AllocationSlot>(addr, nbytes, std::move(deps));
    auto* slot_addr = slot.get();

    if (m_lru_newest != nullptr) {
        m_lru_newest->lru_newer = slot_addr;
        slot->lru_older = m_lru_newest;
        m_lru_newest = slot_addr;
    } else {
        m_lru_newest = slot_addr;
        m_lru_oldest = slot_addr;
    }

    auto& bin = m_allocation_bins[nbytes];
    if (bin.head == nullptr) {
        bin.head = std::move(slot);
        bin.tail = slot_addr;
    } else {
        bin.tail->next = std::move(slot);
        bin.tail = slot_addr;
    }
}

void CachingAllocator::make_progress() {
    m_allocator->make_progress();
}

void CachingAllocator::trim(size_t nbytes_remaining) {
    while (m_bytes_allocated > nbytes_remaining) {
        if (free_some_memory() == 0) {
            break;
        }
    }

    m_allocator->trim(nbytes_remaining);
}

size_t CachingAllocator::free_some_memory() {
    if (m_lru_oldest == nullptr) {
        return 0;
    }

    auto nbytes = m_lru_oldest->nbytes;
    auto& bin = m_allocation_bins[nbytes];

    KMM_ASSERT(bin.head.get() == m_lru_oldest);
    auto slot = std::move(bin.head);

    if (slot->next != nullptr) {
        bin.head = std::move(slot->next);
    } else {
        bin.tail = nullptr;
    }

    KMM_ASSERT(slot->lru_older == nullptr);

    if (auto* newer = slot->lru_newer) {
        newer->lru_older = nullptr;
        m_lru_oldest = newer;
    } else {
        m_lru_oldest = nullptr;
        m_lru_newest = nullptr;
    }

    m_bytes_allocated -= slot->nbytes;
    m_allocator->deallocate_async(slot->addr, slot->nbytes, std::move(slot->dependencies));
    return slot->nbytes;
}

}  // namespace kmm