#pragma once

#include "kmm/core/geometry.hpp"

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
using NDSize = Size<ND_DIMS>;

struct NDRange {
    NDIndex begin;  ///< The starting index of the work chunk.
    NDIndex end;  ///< The ending index of the work chunk.

    /**
     * Initializes an empty chunk.
     */
    KMM_HOST_DEVICE
    NDRange(int64_t x = 1, int64_t y = 1, int64_t z = 1) : begin(0, 0, 0), end(x, y, z) {}

    /**
     * Constructs a chunk with a given begin and end index.
     */
    KMM_HOST_DEVICE
    explicit NDRange(NDIndex begin, NDIndex end) : begin(begin), end(end) {}

    /**
     * Constructs a chunk with a given offset and size.
     */
    KMM_HOST_DEVICE
    explicit NDRange(NDIndex offset, NDSize size) {
        // Doing this in the initializer leads to SEGFAULT in GCC. Why? Don't ask me
        this->begin = offset;
        this->end = offset + size.to_point();
    }

    /**
     * Constructs a range with a given size.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange(Size<N> m) : NDRange(NDSize::from(m)) {}

    /**
     * Constructs a range from an existing range.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange(Range<N> m) : NDRange(NDIndex::from(m.offset), NDSize::from(m.sizes)) {}

    /**
     * Gets the sizes of the work chunk in each dimension.
     */
    KMM_HOST_DEVICE
    NDSize sizes() const {
        return NDSize::from(end - begin);
    }

    /**
     * Gets the size of the work chunk along a specific axis.
     */
    KMM_HOST_DEVICE
    int64_t size(size_t axis) const {
        return axis < ND_DIMS ? end[axis] - begin[axis] : 1;
    }

    /**
     * Gets the total size (volume) of the work chunk.
     */
    KMM_HOST_DEVICE
    int64_t size() const {
        return sizes().volume();
    }

    /**
     * Gets the total size (volume) of the work chunk.
     */
    KMM_HOST_DEVICE
    bool is_empty() const {
        return sizes().is_empty();
    }

    /**
     * Checks if a multidimensional point is contained within the work chunk.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE bool contains(Index<N> p) const {
        bool result = true;

        for (size_t i = 0; i < N && i < ND_DIMS; i++) {
            result &= p[i] >= begin[i] && p[i] < end[i];
        }

        return result;
    }

    /**
     * Checks if a given 3D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t x, int64_t y, int64_t z) const {
        return contains(Index<3> {x, y, z});
    }

    /**
     * Checks if a given 2D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t x, int64_t y) const {
        return contains(Index<2> {x, y});
    }

    /**
     * Checks if a given 1D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t x) const {
        return contains(Index<1> {x});
    }

    /**
     * Returns the effective dimensionality of this range. This the dimensionality `N` such that
     * for all `i >= N` we have `begin[i] == 0` and `end[i] == 1`.
     */
    size_t effective_dimensionality() const {
        for (size_t i = ND_DIMS; i > 0; i--) {
            if (begin[i - 1] == 0 && end[i - 1] == 1) {
                return i;
            }
        }

        return 0;
    }
};

}  // namespace kmm