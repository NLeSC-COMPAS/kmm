#pragma once

#include "checked_compare.hpp"

namespace kmm {

template<typename T>
class Range {
  public:
    using value_type = T;

    constexpr Range(const Range&) = default;
    constexpr Range(Range&&) = default;

    Range& operator=(const Range&) = default;
    Range& operator=(Range&&) = default;

    KMM_HOST_DEVICE
    constexpr Range() : begin(static_cast<T>(0)), end(static_cast<T>(0)) {}

    KMM_HOST_DEVICE
    constexpr Range(T end) : begin(static_cast<T>(0)), end(end) {}

    KMM_HOST_DEVICE
    constexpr Range(T begin, T end) : begin(begin), end(end) {}

    template<typename U>
    KMM_HOST_DEVICE constexpr Range(const Range<U>& that) {
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
    KMM_HOST_DEVICE constexpr bool is_convertible_to() const {
        return is_convertible<U>(begin) && is_convertible<U>(end);
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
    template<typename U = T>
    KMM_HOST_DEVICE constexpr bool contains(const U& index) const {
        return !is_less(index, this->begin) && is_less(index, this->end);
    }

    /**
     * Checks if the given `that` range is fully contained within this range.
     */
    template<typename U = T>
    KMM_HOST_DEVICE constexpr bool contains(const Range<U>& that) const {
        return that.is_empty() ||  //
            (!is_less(that.begin, this->begin) && !is_less(this->end, that.end));
    }

    /**
     * Checks if the given range `that` overlaps this range.
     */
    template<typename U = T>
    KMM_HOST_DEVICE constexpr bool overlaps(const Range<U>& that) const {
        return this->begin < this->end && that.begin < that.end &&  //
            is_less(this->begin, that.end) && is_less(that.begin, this->end);
    }

    /**
     * Returns the range that lies in the intersection of `this` and `that`.
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
    constexpr Range split_tail(T mid) {
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

template<typename L, typename R>
KMM_HOST_DEVICE bool operator==(const Range<L>& lhs, const Range<R>& rhs) {
    return is_equal(lhs.begin, rhs.begin) && is_equal(lhs.end, rhs.end);
}

template<typename L, typename R>
KMM_HOST_DEVICE bool operator!=(const Range<L>& lhs, const Range<R>& rhs) {
    return !(lhs == rhs);
}
}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>
    #include <utility>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {

template<typename T>
std::ostream& operator<<(std::ostream& stream, const Range<T>& p) {
    return stream << p.begin << "..." << p.end;
}

}  // namespace kmm

template<typename T>
struct fmt::formatter<kmm::Range<T>>: fmt::ostream_formatter {};

template<typename T>
struct std::hash<kmm::Range<T>> {
    size_t operator()(const kmm::Range<T>& p) const {
        size_t result = 0;
        kmm::hash_combine(result, p.begin);
        kmm::hash_combine(result, p.end);
        return result;
    }
};
#endif