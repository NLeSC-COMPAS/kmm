#pragma once

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "kmm/runtime/allocators/base.hpp"

namespace kmm {

class BlockAllocator: public AsyncAllocator {
  public:
    static constexpr size_t DEFAULT_BLOCK_SIZE = 1024L * 1024 * 500;

    BlockAllocator(
        std::unique_ptr<AsyncAllocator> allocator,
        size_t min_block_size = DEFAULT_BLOCK_SIZE
    );
    ~BlockAllocator();
    AllocationResult allocate_async(size_t nbytes, void** addr_out, DeviceEventSet& deps_out) final;
    void deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) final;
    void make_progress() final;
    void trim(size_t nbytes_remaining = 0) final;

  private:
    struct Block;
    struct BlockRegion;
    struct BlockRegionSize {
        size_t value;
    };

    struct RegionSizeCompare {
        using is_transparent = void;
        bool operator()(const BlockRegion*, const BlockRegion*) const;
        bool operator()(const BlockRegion*, BlockRegionSize) const;
    };

    BlockRegion* allocate_block(size_t min_nbytes);
    BlockRegion* find_region(size_t nbytes, size_t alignment);
    static std::pair<BlockRegion*, BlockRegion*> split_region(
        BlockRegion* region,
        size_t left_size
    );
    static BlockRegion* merge_regions(BlockRegion* left, BlockRegion* right);
    static size_t offset_to_alignment(const BlockRegion* region, size_t alignment);
    static bool fits_in_region(const BlockRegion* region, size_t nbytes, size_t alignment);

    std::unique_ptr<AsyncAllocator> m_allocator;
    std::unordered_map<void*, BlockRegion*> m_active_regions;
    std::vector<std::unique_ptr<Block>> m_blocks;
    size_t m_active_block = 0;
    size_t m_min_block_size = DEFAULT_BLOCK_SIZE;
    size_t m_bytes_allocated = 0;
};
}  // namespace kmm