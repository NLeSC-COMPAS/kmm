#pragma once

#include "kmm/core/geometry.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

/**
 * Constant for the number of dimensions in the work space.
 */
static constexpr size_t ND_DIMS = 3;

/**
 * Type alias for the index type used in the work space.
 */
using NDIndex = Index<ND_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using NDSize = Dim<ND_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using NDRange = Bounds<ND_DIMS>;

}  // namespace kmm