#pragma once

#include <cstddef>
#include <vector>

#include "kmm/core/identifiers.hpp"
#include "kmm/core/resource.hpp"
#include "kmm/core/system_info.hpp"
#include "kmm/utils/geometry.hpp"

namespace kmm {

/**
 * Constant for the number of dimensions in the work space.
 */
static constexpr size_t DOMAIN_DIMS = 3;

/**
 * Type alias for the index type used in the work space.
 */
using DomainIndex = Index<DOMAIN_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using DomainDim = Dim<DOMAIN_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using DomainBounds = Bounds<DOMAIN_DIMS>;

struct DomainChunk {
    ResourceId owner_id;
    DomainIndex offset;
    DomainDim size;
};

struct Domain {
    std::vector<DomainChunk> chunks;
};

template<typename P>
struct IntoDomain {
    static Domain call(P partition, const SystemInfo& info, ExecutionSpace space) {
        return partition;
    }
};

struct TileDomain {
    TileDomain(DomainDim domain_size, DomainDim tile_size) :
        m_domain_size(domain_size),
        m_tile_size(tile_size) {}

    Domain operator()(const SystemInfo& info, ExecutionSpace space) const;

  private:
    DomainBounds m_domain_size;
    DomainDim m_tile_size;
};

template<>
struct IntoDomain<TileDomain> {
    static Domain call(TileDomain partition, const SystemInfo& info, ExecutionSpace space) {
        return partition(info, space);
    }
};

}  // namespace kmm
