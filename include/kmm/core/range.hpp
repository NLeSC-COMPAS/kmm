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
 * A one-dimensional range from `begin` up to, but not included, `end`.
 */
template<typename T = default_index_type>
struct Range {
    using value_type = default_index_type;

    value_type begin {};
    value_type end {};

    constexpr Range() = default;
    constexpr Range(const Range& that) = default;

    KMM_HOST_DEVICE
    constexpr Range(value_type end) : Range(value_type {}, end) {}

    KMM_HOST_DEVICE
    constexpr Range(value_type begin, value_type end) : begin(begin), end(end) {}

    template<typename U>
    explicit constexpr Range(const Range<U>& that) :
        Range(static_cast<T>(that.begin), static_cast<T>(that.end)) {}

    /**
     * Checks if the given index `index` is within this range.
     */
    KMM_HOST_DEVICE
    constexpr bool contains(const value_type& index) const {
        return index >= this->begin && index < this->end;
    }

    /**
     * Checks if the given `that` range is fully contained within this range.
     */
    KMM_HOST_DEVICE
    constexpr bool contains(const Range& that) const {
        return that.begin >= this->begin && that.end <= this->end;
    }

    /**
     * Checks if the given `that` range overlaps this range.
     */
    KMM_HOST_DEVICE
    constexpr bool overlaps(const Range& that) const {
        return this->begin < that.end && that.begin < this->end;
    }

    /**
     * Computes the size (or length) of the range.
     */
    KMM_HOST_DEVICE
    constexpr value_type size() const {
        return this->begin <= this->end ? this->end - this->begin : value_type {0};
    }

    /**
     * Checks if the range is empty (i.e., `begin == end`) or invalid (i.e., `begin > end`).
     */
    KMM_HOST_DEVICE
    constexpr bool is_empty() const {
        return this->begin >= this->end;
    }

    /**
     * Returns the range `mid...end` and modifies the current range such it becomes `begin...mid`.
     */
    KMM_HOST_DEVICE
    constexpr Range split_at(T mid) {
        auto old_end = this->end;
        this->end = mid;
        return {mid, old_end};
    }

    /**
     * Shift the range by the given amount.
     */
    KMM_HOST_DEVICE
    constexpr Range shift_by(T shift) {
        return {begin + shift, end + shift};
    }
};

struct NDRange: fixed_array<Range<default_index_type>, ND_DIMS> {
    using value_type = default_index_type;

    /**
     * Initializes from range objects
     */
    KMM_HOST_DEVICE
    NDRange(Range<value_type> x, Range<value_type> y = 1, Range<value_type> z = 1) :
        fixed_array<Range<default_index_type>, ND_DIMS> {x, y, z} {}

    /**
     * Initializes from sizes along each axis.
     */
    KMM_HOST_DEVICE
    NDRange(value_type x = 1, value_type y = 1, value_type z = 1) :
        NDRange(Range<value_type>(x), Range<value_type>(y), Range<value_type>(z)) {}

    /**
     * Constructs a range with a given size.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange(Dim<N> size) : NDRange(size.get(0), size.get(1), size.get(2)) {}

    /**
     * Constructs a range with a given size.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange(Index<N> end) {
        *this = NDRange(Dim<N>::from(end));
    }

    /**
     * Constructs a range from the bounds.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange(Bounds<N> m) {
        *this = from_bounds(m.begin, m.end);
    }

    /**
     * Initialize from a given begin and end index.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE static NDRange from_bounds(Index<N> begin, Index<N> end) {
        return NDRange {
            N > 0 ? Range {begin[0], end[0]} : 1,
            N > 1 ? Range {begin[1], end[1]} : 1,
            N > 2 ? Range {begin[2], end[2]} : 1,
        };
    }

    /**
     * Initialize from a given offset and size.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE static NDRange from_offset_size(Index<N> offset, Dim<N> size) {
        return NDRange(size).shift_by(offset);
    }

    /**
     * Get the range for the specified axis.
     */
    KMM_HOST_DEVICE
    Range<value_type> get(size_t axis) const {
        return KMM_LIKELY(axis < ND_DIMS) ? (*this)[axis] : 1;
    }

    /**
     * Gets the starting coordinate of the specified axis.
     */
    KMM_HOST_DEVICE
    value_type begin(size_t axis) const {
        return get(axis).begin;
    }

    /**
     * Gets the ending coordinate of the specified axis.
     */
    KMM_HOST_DEVICE
    value_type end(size_t axis) const {
        return get(axis).end;
    }

    /**
     * Gets the size (length) of the specified axis.
     */
    KMM_HOST_DEVICE
    value_type size(size_t axis) const {
        return get(axis).size();
    }

    /**
     * Gets the total size (volume).
     */
    KMM_HOST_DEVICE
    value_type size() const {
        return x.size() * y.size() * z.size();
    }

    /**
     * Checks if the range is empty in any of the dimensions.
     */
    KMM_HOST_DEVICE
    bool is_empty() const {
        return x.is_empty() || y.is_empty() || z.is_empty();
    }

    /**
     * Returns an `Index<N>` representing the beginning of each axis.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE Index<N> begin() const {
        auto result = Index<N>::zero();

        for (size_t i = 0; i < N; i++) {
            result[i] = get(i).begin;
        }

        return result;
    }

    /**
     * Returns an `Index<N>` representing the ending of each axis.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE Index<N> end() const {
        auto result = Index<N>::one();

        for (size_t i = 0; i < N; i++) {
            result[i] = get(i).end;
        }

        return result;
    }

    /**
     * Returns an `Size<N>` of the sizes of each axis.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE Dim<N> sizes() const {
        auto result = Dim<N>::one();

        for (size_t i = 0; i < N && i < ND_DIMS; i++) {
            result[i] = get(i).size();
        }

        return result;
    }

    /**
     * Checks if a multidimensional point is contained within this range.
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
     * Checks if a given 3D point is contained within the range.
     */
    KMM_HOST_DEVICE
    bool contains(value_type px, value_type py, value_type pz) const {
        return contains(Index<3> {px, py, pz});
    }

    /**
     * Checks if a given 2D point is contained within the range.
     */
    KMM_HOST_DEVICE
    bool contains(value_type px, value_type py) const {
        return contains(Index<2> {px, py});
    }

    /**
     * Checks if a given 1D point is contained within the range.
     */
    KMM_HOST_DEVICE
    bool contains(value_type px) const {
        return contains(Index<1> {px});
    }

    /**
     * Split the range along the given axis. modifies the current object along the given axis such
     * it becomes `begin...mid`. Returns a copy of the current object where the range at the
     * given axis is replaced by `mid..end`.
     */
    KMM_HOST_DEVICE
    NDRange split_along(size_t axis, value_type mid) {
        if (axis < ND_DIMS) {
            auto result = *this;
            result[axis] = (*this)[axis].split_at(mid);
            return result;
        } else {
            return 0;
        }
    }

    /**
     * Shift the range by the given amount.
     */
    template<size_t N = ND_DIMS>
    KMM_HOST_DEVICE NDRange shift_by(Index<N> amount) {
        return {
            x.shift_by(amount.get(0)),
            y.shift_by(amount.get(1)),
            z.shift_by(amount.get(2)),
        };
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