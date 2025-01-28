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

template<typename T = default_geometry_type>
struct Range {
    T begin;
    T end;

    KMM_HOST_DEVICE
    Range() : begin(T {}), end(T {}) {}

    KMM_HOST_DEVICE
    Range(T end) : begin(T {}), end(end) {}

    KMM_HOST_DEVICE
    Range(T begin, T end) : begin(begin), end(end) {}

    template<typename U>
    KMM_HOST_DEVICE Range(const Range<U>& that) : begin(that.begin), end(that.end) {}

    template<typename U>
    KMM_HOST_DEVICE Range(Size<1, U> that) : end(that[0]) {}

    template<typename U>
    KMM_HOST_DEVICE Range(Bounds<1, U> that) : begin(that.begin()), end(that.end()) {}

    KMM_HOST_DEVICE
    T contains(const T& index) const {
        return index >= begin && index < end;
    }

    KMM_HOST_DEVICE
    T contains(const Range<T>& that) const {
        return that.begin >= begin && that.end < end;
    }

    KMM_HOST_DEVICE
    T size() const {
        return begin <= end ? end - begin : T {0};
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        return begin >= end;
    }
};

struct NDRange {
    Range<default_geometry_type> x;
    Range<default_geometry_type> y;
    Range<default_geometry_type> z;

    /**
     * Initializes from sizes.
     */
    KMM_HOST_DEVICE
    NDRange(default_geometry_type x = 1, default_geometry_type y = 1, default_geometry_type z = 1) :
        x(x),
        y(y),
        z(z) {}

    /**
     * Initializes from a set of ranges.
     */
    KMM_HOST_DEVICE
    NDRange(
        Range<default_geometry_type> x,
        Range<default_geometry_type> y = 1,
        Range<default_geometry_type> z = 1
    ) :
        x(x),
        y(y),
        z(z) {}

    /**
     * Constructs a chunk with a given begin and end index.
     */
    KMM_HOST_DEVICE
    explicit NDRange(NDIndex begin, NDIndex end) :
        x(begin.x, end.x),
        y(begin.y, end.y),
        z(begin.z, end.z) {}

    /**
     * Constructs a chunk with a given offset and size.
     */
    KMM_HOST_DEVICE
    explicit NDRange(NDIndex offset, NDSize size) {
        // Doing this in the initializer leads to SEGFAULT in GCC. Why? Don't ask me
        *this = NDRange(offset, offset + size.to_point());
    }

    /**
     * Constructs a range with a given size.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange(Size<N> m) {
        *this = NDRange(NDIndex::zero(), NDSize::from(m));
    }

    /**
     * Constructs a range from the bounds.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange(Bounds<N> m) {
        *this = NDRange(NDIndex::from(m.begin()), NDSize::from(m.end()));
    }

    /**
     * Get the range along the given index
     */
    KMM_HOST_DEVICE
    Range<> get(size_t axis) const {
        switch (axis) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                return 1;
        }
    }

    /**
     * Get the range along the given index
     */
    KMM_HOST_DEVICE
    Range<> operator[](size_t axis) const {
        return get(axis);
    }

    KMM_HOST_DEVICE
    int64_t begin(size_t axis) const {
        return get(axis).begin;
    }

    KMM_HOST_DEVICE
    int64_t end(size_t axis) const {
        return get(axis).end;
    }

    /**
     * Gets the size of the work chunk along a specific axis.
     */
    KMM_HOST_DEVICE
    int64_t size(size_t axis) const {
        return get(axis).size();
    }

    /**
     * Gets the total size (volume) of the work chunk.
     */
    KMM_HOST_DEVICE
    int64_t size() const {
        return x.size() * y.size() * z.size();
    }

    /**
     * Gets the total size (volume) of the work chunk.
     */
    KMM_HOST_DEVICE
    bool is_empty() const {
        return x.is_empty() || y.is_empty() || z.is_empty();
    }

    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE Index<N> begin() const {
        auto result = Index<N>::zero();

        for (size_t i = 0; i < N; i++) {
            result[i] = get(i).begin;
        }

        return result;
    }

    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE Index<N> end() const {
        auto result = Index<N>::one();

        for (size_t i = 0; i < N; i++) {
            result[i] = get(i).end;
        }

        return result;
    }

    /**
     * Gets the sizes of the work chunk in each dimension.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE Size<N> sizes() const {
        auto result = Size<N>::one();

        for (size_t i = 0; i < N && i < ND_DIMS; i++) {
            result[i] = get(i).size();
        }

        return result;
    }

    /**
     * Checks if a multidimensional point is contained within the work chunk.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE bool contains(Index<N> p) const {
        bool result = true;

        for (size_t i = 0; i < N && i < ND_DIMS; i++) {
            result &= get(i).contains(p[i]);
        }

        return result;
    }

    /**
     * Checks if a given 3D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t px, int64_t py, int64_t pz) const {
        return contains(Index<3> {px, py, pz});
    }

    /**
     * Checks if a given 2D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t px, int64_t py) const {
        return contains(Index<2> {px, py});
    }

    /**
     * Checks if a given 1D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t px) const {
        return contains(Index<1> {px});
    }

    /**
     * Returns the effective dimensionality of this range. This the dimensionality `N` such that
     * for all `i >= N` we have `begin[i] == 0` and `end[i] == 1`.
     */
    size_t effective_dimensionality() const {
        for (size_t i = ND_DIMS; i > 0; i--) {
            if (get(i - 1).begin == 0 && get(i - 1).end == 1) {
                return i;
            }
        }

        return 0;
    }
};

}  // namespace kmm