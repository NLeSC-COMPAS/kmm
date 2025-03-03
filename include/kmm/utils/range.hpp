#pragma once

#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

using default_index_type = int64_t;

template<typename T = default_index_type>
class Range {
  public:
    using value_type = T;

    constexpr Range(const Range&) = default;
    constexpr Range(Range&&) = default;
    Range& operator=(const Range&) = default;
    Range& operator=(Range&&) = default;

    KMM_HOST_DEVICE
    constexpr Range() : begin(T {}), end(static_cast<T>(1)) {}

    KMM_HOST_DEVICE
    constexpr Range(T end) : begin(T {}), end(end) {}

    KMM_HOST_DEVICE
    constexpr Range(T begin, T end) : begin(begin), end(end) {}

    template<typename U>
    constexpr Range(const Range<U>& that) {
        if (!that.template is_convertible_to<T>()) {
            throw_overflow_exception();
        }

        *this = Range::from(that);
    }

    template<typename U = T>
    KMM_HOST_DEVICE static Range from(const Range<U>& range) {
        return {static_cast<T>(range.begin), static_cast<T>(range.end)};
    }

    template<typename U>
    constexpr bool is_convertible_to() const {
        return in_range<U>(begin) && in_range<U>(end);
    }

    /**
     * Checks if the range is empty (i.e., `begin == end`) or invalid (i.e., `begin > end`).
     */
    KMM_HOST_DEVICE
    constexpr bool is_empty() const {
        return this->begin >= this->end;
    }

    /**
     * Checks if the given index `index` is within this range.
     */
    KMM_HOST_DEVICE
    constexpr bool contains(const T& index) const {
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
     * Checks if the given range `that` overlaps this range.
     */
    KMM_HOST_DEVICE
    constexpr bool overlaps(const Range& that) const {
        return this->begin < that.end && that.begin < this->end;
    }

    /**
     * Checks if the given range `that` overlaps this range.
     */
    KMM_HOST_DEVICE
    constexpr Range intersection(const Range& that) const {
        return {
            this->begin > that.begin ? this->begin : that.begin,
            this->end < that.end ? this->end : that.end,
        };
    }

    /**
     * Computes the size (or length) of the range.
     */
    KMM_HOST_DEVICE
    constexpr T size() const {
        return this->begin <= this->end ? this->end - this->begin : static_cast<T>(0);
    }

    /**
     * Returns the range `mid...end` and modifies the current range such it becomes `begin...mid`.
     */
    KMM_HOST_DEVICE
    constexpr Range split_off(T mid) {
        if (mid < this->begin) {
            mid = this->begin;
        }

        if (mid > this->end) {
            mid = this->end;
        }

        auto old_end = this->end;
        this->end = mid;
        return {mid, old_end};
    }

    /**
     * Returns a new range that has been shifted by the given amount.
     */
    KMM_HOST_DEVICE
    constexpr Range shift_by(T shift) const {
        return {this->begin + shift, this->end + shift};
    }

    T begin;
    T end;
};

template<typename T>
Range(const T&) -> Range<T>;

template<typename T>
Range(const T&, const T&) -> Range<T>;

template<typename T>
KMM_HOST_DEVICE bool operator==(const Range<T>& lhs, const Range<T>& rhs) {
    return lhs.begin == rhs.begin && lhs.end == rhs.end;
}

template<typename T>
KMM_HOST_DEVICE bool operator!=(const Range<T>& lhs, const Range<T>& rhs) {
    return !(lhs == rhs);
}

}  // namespace kmm

#include <iosfwd>

namespace kmm {

template<typename T>
std::ostream& operator<<(std::ostream& stream, const Range<T>& p) {
    return stream << p.begin << "..." << p.end;
}

}  // namespace kmm

#include "kmm/utils/hash_utils.hpp"

template<typename T>
struct std::hash<kmm::Range<T>> {
    size_t operator()(const kmm::Range<T>& p) const {
        size_t result = 0;
        kmm::hash_combine(result, p.begin);
        kmm::hash_combine(result, p.end);
        return result;
    }
};