#pragma once

#include "kmm/core/geometry.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

/**
 * Constant for the number of dimensions in the work space.
 */
static constexpr size_t WORK_DIMS = 3;

/**
 * Type alias for the index type used in the work space.
 */
using WorkIndex = Index<WORK_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using WorkDim = Dim<WORK_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using WorkBounds = Bounds<WORK_DIMS>;

}  // namespace kmm