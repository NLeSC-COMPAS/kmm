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
class DomainPoint: public Point<DOMAIN_DIMS> {
  public:
    DomainPoint(Point<0> p) : Point<DOMAIN_DIMS>(p) {}
    DomainPoint(Point<1> p) : Point<DOMAIN_DIMS>(p) {}
    DomainPoint(Point<2> p) : Point<DOMAIN_DIMS>(p) {}
    DomainPoint(Point<3> p) : Point<DOMAIN_DIMS>(p) {}

    DomainPoint(
        default_index_type x = 0,  //
        default_index_type y = 0,
        default_index_type z = 0
    ) :
        Point<DOMAIN_DIMS>(x, y, z) {}
};

/**
 * Type alias for the size of the work space.
 */
class DomainDim: public Dim<DOMAIN_DIMS> {
  public:
    DomainDim(Dim<0> p) : Dim<DOMAIN_DIMS>(p) {}
    DomainDim(Dim<1> p) : Dim<DOMAIN_DIMS>(p) {}
    DomainDim(Dim<2> p) : Dim<DOMAIN_DIMS>(p) {}
    DomainDim(Dim<3> p) : Dim<DOMAIN_DIMS>(p) {}

    DomainDim(
        default_index_type x = 1,  //
        default_index_type y = 1,
        default_index_type z = 1
    ) :
        Dim<DOMAIN_DIMS>(x, y, z) {}
};

/**
 * Type alias for the size of the work space.
 */
class DomainBounds: public Bounds<DOMAIN_DIMS> {
  public:
    DomainBounds(Bounds<0> p) : Bounds<DOMAIN_DIMS>(p) {}
    DomainBounds(Bounds<1> p) : Bounds<DOMAIN_DIMS>(p) {}
    DomainBounds(Bounds<2> p) : Bounds<DOMAIN_DIMS>(p) {}
    DomainBounds(Bounds<3> p) : Bounds<DOMAIN_DIMS>(p) {}

    DomainBounds(Dim<0> p) : Bounds<DOMAIN_DIMS>(p) {}
    DomainBounds(Dim<1> p) : Bounds<DOMAIN_DIMS>(p) {}
    DomainBounds(Dim<2> p) : Bounds<DOMAIN_DIMS>(p) {}
    DomainBounds(Dim<3> p) : Bounds<DOMAIN_DIMS>(p) {}

    DomainBounds(
        Range<default_index_type> x = 1,
        Range<default_index_type> y = 1,
        Range<default_index_type> z = 1
    ) :
        Bounds<DOMAIN_DIMS>(x, y, z) {}
};

struct DomainChunk {
    ResourceId owner_id;
    DomainPoint offset;
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

template<>
struct fmt::formatter<kmm::DomainPoint>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::DomainDim>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::DomainBounds>: fmt::ostream_formatter {};