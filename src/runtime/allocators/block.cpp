#include "kmm/runtime/allocators/block.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

static constexpr size_t MAX_ALIGNMENT = 256;

struct BlockAllocator::Block {
    std::set<BlockRegion*, RegionSizeCompare> free_regions;
    std::unique_ptr<BlockRegion> head = nullptr;
    BlockRegion* tail = nullptr;
    void* base_addr = nullptr;
    size_t size = 0;

    Block(void* addr, size_t size, DeviceEventSet deps) {
        this->head = std::make_unique<BlockRegion>(this, 0, size, (deps));
        this->tail = head.get();
        this->free_regions.insert(this->head.get());
        this->base_addr = addr;
        this->size = size;
    }
};

struct BlockAllocator::BlockRegion {
    Block* parent = nullptr;
    std::unique_ptr<BlockRegion> next = nullptr;
    BlockRegion* prev = nullptr;
    size_t offset_in_block = 0;
    size_t size = 0;
    bool is_free = false;
    DeviceEventSet dependencies;

    BlockRegion(Block* parent, size_t offset_in_block, size_t size, DeviceEventSet deps = {}) {
        this->parent = parent;
        this->offset_in_block = offset_in_block;
        this->size = size;
        this->dependencies = (deps);
    }
};

bool BlockAllocator::RegionSizeCompare::operator()(const BlockRegion* a, const BlockRegion* b)
    const {
    return a->size < b->size;
}

bool BlockAllocator::RegionSizeCompare::operator()(const BlockRegion* a, BlockRegionSize b) const {
    return a->size < b.value;
}

BlockAllocator::BlockAllocator(std::unique_ptr<AsyncAllocator> allocator, size_t min_block_size) :
    m_allocator(std::move(allocator)),
    m_min_block_size(min_block_size) {}

BlockAllocator::~BlockAllocator() {
    for (size_t index = 0; index < m_blocks.size(); index++) {
        auto& block = m_blocks[index];
        auto* region = m_blocks[index]->head.get();

        if (!region->is_free || region->next != nullptr) {
            // OH NO
            continue;
        }

        m_allocator->deallocate_async(  //
            block->base_addr,
            block->size,
            std::move(region->dependencies)
        );
    }
}

AllocationResult BlockAllocator::allocate_async(
    size_t nbytes,
    void** addr_out,
    DeviceEventSet& deps_out
) {
    size_t alignment = std::min(round_up_to_power_of_two(nbytes), MAX_ALIGNMENT);
    nbytes = round_up_to_multiple(nbytes, alignment);

    auto* region = find_region(nbytes, alignment);

    if (region == nullptr) {
        region = allocate_block(nbytes);

        if (region == nullptr) {
            return AllocationResult::ErrorOutOfMemory;
        }
    }

    auto* block = region->parent;
    block->free_regions.erase(region);
    region->is_free = false;

    auto offset_in_region = offset_to_alignment(region, alignment);

    if (region->size > offset_in_region + nbytes) {
        auto [left, right] = split_region(region, offset_in_region + nbytes);
        block->free_regions.emplace(right);
        region = left;
    }

    *addr_out = static_cast<char*>(block->base_addr) + region->offset_in_block + offset_in_region;
    deps_out.insert(region->dependencies);

    m_active_regions.emplace(addr_out, region);
    return AllocationResult::Success;
}

BlockAllocator::BlockRegion* BlockAllocator::allocate_block(size_t min_nbytes) {
    DeviceEventSet deps;
    void* base_addr;
    size_t block_size = std::max(min_nbytes, m_min_block_size);

    while (true) {
        auto result = m_allocator->allocate_async(block_size, &base_addr, deps);

        if (result == AllocationResult::Success) {
            break;
        }

        block_size /= 2;

        if (block_size < min_nbytes) {
            return nullptr;
        }
    }

    auto new_block = std::make_unique<Block>(base_addr, block_size, std::move(deps));
    auto* region = new_block->head.get();

    m_bytes_allocated += block_size;
    m_blocks.insert(
        m_blocks.begin() + static_cast<ptrdiff_t>(m_active_block),
        std::move(new_block)
    );

    return region;
}

BlockAllocator::BlockRegion* BlockAllocator::find_region(size_t nbytes, size_t alignment) {
    for (size_t i = 0; i < m_blocks.size(); i++) {
        auto* block = m_blocks[m_active_block].get();
        auto it = block->free_regions.lower_bound(BlockRegionSize {nbytes});

        while (it != block->free_regions.end()) {
            auto* region = &**it;
            it++;

            if (fits_in_region(region, nbytes, alignment)) {
                return region;
            }
        }

        m_active_block = (m_active_block + 1) % m_blocks.size();
    }

    return nullptr;
}

void BlockAllocator::deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) {
    auto it = m_active_regions.find(addr);
    KMM_ASSERT(it != m_active_regions.end());

    auto* region = it->second;
    m_active_regions.erase(it);

    KMM_ASSERT(nbytes <= region->size);
    KMM_ASSERT(region->is_free == false);

    region->is_free = true;
    region->dependencies = std::move(deps);

    auto* block = region->parent;
    auto* prev = region->prev;
    auto* next = region->next.get();

    if (prev != nullptr && prev->is_free) {
        block->free_regions.erase(prev);
        region = merge_regions(prev, region);
    }

    if (next != nullptr && next->is_free) {
        block->free_regions.erase(next);
        region = merge_regions(region, next);
    }

    block->free_regions.insert(region);
}

size_t BlockAllocator::offset_to_alignment(const BlockRegion* region, size_t alignment) {
    return round_up_to_multiple(region->offset_in_block, alignment) - region->offset_in_block;
}

bool BlockAllocator::fits_in_region(const BlockRegion* region, size_t nbytes, size_t alignment) {
    return region->size >= offset_to_alignment(region, alignment) + nbytes;
}

auto BlockAllocator::split_region(BlockRegion* region, size_t left_size)
    -> std::pair<BlockRegion*, BlockRegion*> {
    KMM_ASSERT(region->size > left_size);

    auto* parent = region->parent;
    auto* left = region;

    size_t right_offset = left->offset_in_block + left_size;
    size_t right_size = left->size - left_size;
    left->size = left_size;

    auto right =
        std::make_unique<BlockRegion>(parent, right_offset, right_size, left->dependencies);
    auto* right_ptr = right.get();

    if (left->next != nullptr) {
        left->next->prev = right_ptr;
        right->next = std::move(left->next);
    } else {
        parent->tail = right_ptr;
        right->next = nullptr;
    }

    right->prev = left;
    left->next = std::move(right);

    return {left, right_ptr};
}

auto BlockAllocator::merge_regions(BlockRegion* left, BlockRegion* right) -> BlockRegion* {
    auto* parent = left->parent;

    KMM_ASSERT(left->parent == parent && right->parent == parent);
    KMM_ASSERT(left->is_free && right->is_free);
    KMM_ASSERT(left->next.get() == right);
    KMM_ASSERT(left == right->prev);

    if (right->next != nullptr) {
        right->next->prev = left;
    } else {
        parent->tail = left;
    }

    left->size += right->size;
    left->dependencies.insert(right->dependencies);
    left->next = std::move(right->next);  // `right` is deleted here (since left.next == right)
    return left;
}

void BlockAllocator::make_progress() {
    m_allocator->make_progress();
}

void BlockAllocator::trim(size_t nbytes_remaining) {
    size_t index = 0;

    while (m_bytes_allocated >= nbytes_remaining) {
        if (index >= m_blocks.size()) {
            break;
        }

        auto& block = m_blocks[index];
        auto* region = block->head.get();

        if (!region->is_free || region->next != nullptr) {
            index++;
            continue;
        }

        m_allocator->deallocate_async(  //
            block->base_addr,
            block->size,
            std::move(region->dependencies)
        );

        m_bytes_allocated -= block->size;
        m_blocks.erase(m_blocks.begin() + static_cast<ptrdiff_t>(index));

        if (m_active_block > index) {
            m_active_block--;
        }
    }

    m_allocator->trim(nbytes_remaining);
}

}  // namespace kmm
