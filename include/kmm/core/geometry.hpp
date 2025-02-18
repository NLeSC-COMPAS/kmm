#pragma once

#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/fixed_array.hpp"

namespace kmm {

using default_index_type = int64_t;

template<size_t N, typename T = default_index_type>
class Index: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Index(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Index() {
        for (size_t i = 0; is_less(i, N); i++) {
            (*this)[i] = T {};
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Index(T first, Ts&&... args) : Index() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    constexpr Index(const Index&) = default;
    constexpr Index(Index&&) noexcept = default;
    Index& operator=(const Index&) = default;
    Index& operator=(Index&&) noexcept = default;

    template<size_t M, typename U>
    constexpr Index(const Index<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Index::from(that);
    }

    KMM_HOST_DEVICE
    static Index fill(T value) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = value;
        }

        return Index(result);
    }

    KMM_HOST_DEVICE
    static Index one() {
        return fill(static_cast<T>(1));
    }

    KMM_HOST_DEVICE
    static Index zero() {
        return fill(static_cast<T>(0));
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Index from(const fixed_array<U, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = i < M ? static_cast<T>(that[i]) : static_cast<T>(0);
        }

        return Index(result);
    }

    KMM_HOST_DEVICE
    T get_or_default(size_t i, T default_value = {}) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(i < N)) {
                return (*this)[i];
            }
        }

        return default_value;
    }

    template<size_t M = N, typename U = T>
    bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (is_less(i, M)) {
                result &= in_range<U>((*this)[i]);
            } else {
                result &= compare_equal((*this)[i], static_cast<T>(0));
            }
        }

        return result;
    }
};

template<size_t N, typename T = default_index_type>
struct Dim: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Dim(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Dim() {
        for (size_t i = 0; is_less(i, N); i++) {
            (*this)[i] = static_cast<T>(1);
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Dim(T first, Ts&&... args) : Dim() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    template<size_t M, typename U>
    constexpr Dim(const Dim<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Dim::from(that);
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Dim from(const fixed_array<U, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? static_cast<T>(that[i]) : static_cast<T>(1);
        }

        return Dim(result);
    }

    KMM_HOST_DEVICE
    static Dim fill(T value) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = value;
        }

        return Dim(result);
    }

    KMM_HOST_DEVICE
    static Dim one() {
        return fill(static_cast<T>(1));
    }

    KMM_HOST_DEVICE
    static Dim zero() {
        return fill(static_cast<T>(0));
    }

    KMM_HOST_DEVICE
    T get_or_default(size_t i, T default_value = static_cast<T>(1)) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(i < N)) {
                return (*this)[i];
            }
        }

        return default_value;
    }

    template<size_t M = N, typename U = T>
    bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (i < M) {
                result &= in_range<U>((*this)[i]);
            } else {
                result &= compare_equal((*this)[i], static_cast<T>(0));
            }
        }

        return result;
    }

    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= !(static_cast<T>(0) < (*this)[i]);
        }

        return result;
    }

    T volume() const {
        T result = static_cast<T>(1);

        if constexpr (N >= 1) {
            result = (*this)[0];

            for (size_t i = 1; is_less(i, N); i++) {
                result *= (*this)[i];
            }
        }

        return is_empty() ? static_cast<T>(0) : result;
    }

    bool contains(const Index<N, T>& p) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= p[i] >= 0 && p[i] < (*this)[i];
        }

        return result;
    }

    bool contains(const Dim<N, T>& p) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= p[i] <= (*this)[i];
        }

        return p.is_empty() || result;
    }

    KMM_HOST_DEVICE
    Dim intersection(const Dim& that) const {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            if ((*this)[i] < 0 || that[i] < 0) {
                result[i] = static_cast<T>(0);
            } else if ((*this)[i] <= that[i]) {
                result[i] = (*this)[i];
            } else {
                result[i] = that[i];
            }
        }

        return Dim(result);
    }

    KMM_HOST_DEVICE
    bool overlaps(const Dim& that) const {
        return !this->is_empty() && !that.is_empty();
    }
};

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
    KMM_HOST_DEVICE constexpr bool is_convertible_to() const {
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
     * Checks if the given `that` range overlaps this range.
     */
    KMM_HOST_DEVICE
    constexpr bool overlaps(const Range& that) const {
        return this->begin < that.end && that.begin < this->end;
    }

    /**
     * Checks if the given `that` range overlaps this range.
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
     * Shift the range by the given amount.
     */
    KMM_HOST_DEVICE
    constexpr Range shift_by(T shift) {
        return {this->begin + shift, this->end + shift};
    }

    T begin;
    T end;
};

template<typename T>
KMM_HOST_DEVICE bool operator==(const Range<T>& lhs, const Range<T>& rhs) {
    return lhs.begin == rhs.begin && lhs.end == rhs.end;
}

template<typename T>
KMM_HOST_DEVICE bool operator!=(const Range<T>& lhs, const Range<T>& rhs) {
    return !(lhs == rhs);
}

template<size_t N, typename T = default_index_type>
class Bounds: public fixed_array<Range<T>, N> {
  public:
    using storage_type = fixed_array<Range<T>, N>;

    KMM_HOST_DEVICE
    explicit constexpr Bounds(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    Bounds() {
        for (size_t i = 0; is_less(i, N); i++) {
            (*this)[i] = Range<T>();
        }
    }

    constexpr Bounds(const Bounds&) = default;
    constexpr Bounds(Bounds&&) noexcept = default;
    Bounds& operator=(const Bounds&) = default;
    Bounds& operator=(Bounds&&) noexcept = default;

    template<size_t M, typename U>
    constexpr Bounds(const Bounds<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Bounds::from(that);
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Bounds(Range<T> first, Ts&&... args) : Bounds() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    template<size_t K, typename = typename std::enable_if<(K <= N)>::type>
    KMM_HOST_DEVICE Bounds(const Dim<K, T>& shape) {
        *this = from_offset_size(Index<N, T>::zero(), Dim<N, T>::from(shape));
    }

    KMM_HOST_DEVICE static constexpr Bounds from_bounds(
        const Index<N, T>& begin,
        const Index<N, T>& end
    ) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = {begin[i], end[i]};
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE static constexpr Bounds from_offset_size(
        const Index<N, T>& offset,
        const Dim<N, T>& shape
    ) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = Range<T>(shape[i]).shift_by(offset[i]);
        }

        return Bounds(result);
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Bounds from(const fixed_array<Range<U>, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? Range<T>::from(that[i]) : Range<T>();
        }

        return Bounds(result);
    }

    template<size_t M = N, typename U = T>
    bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (i < M) {
                result &= (*this)[i].template is_convertible_to<U>();
            } else {
                result &= (*this)[i] == Range<T>();
            }
        }

        return result;
    }

    KMM_HOST_DEVICE
    Range<T> get_or_default(size_t i, Range<T> default_value = {}) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(i < N)) {
                return (*this)[i];
            }
        }

        return default_value;
    }

    KMM_HOST_DEVICE
    T begin(size_t axis) const {
        return (*this)[axis].begin;
    }

    KMM_HOST_DEVICE
    T end(size_t axis) const {
        return (*this)[axis].end;
    }

    KMM_HOST_DEVICE
    T size(size_t axis) const {
        return (*this)[axis].size();
    }

    KMM_HOST_DEVICE
    Index<N, T> begin() const {
        Index<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = this->begin(axis);
        }
        return result;
    }

    KMM_HOST_DEVICE
    Index<N, T> end() const {
        Index<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = this->end(axis);
        }
        return result;
    }

    KMM_HOST_DEVICE
    Dim<N, T> sizes() const {
        Dim<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = this->size(axis);
        }
        return result;
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= begin(i) >= end(i);
        }

        return result;
    }

    KMM_HOST_DEVICE
    T size() const {
        T result = 1;

        for (size_t i = 0; is_less(i, N); i++) {
            result *= this->end(i) - this->begin(i);
        }

        return this->is_empty() ? T {0} : result;
    }

    KMM_HOST_DEVICE
    Bounds intersection(const Bounds& that) const {
        Index<N, T> new_begin;
        Index<N, T> new_end;

        for (size_t i = 0; is_less(i, N); i++) {
            new_begin[i] = this->begin(i) >= that.begin(i) ? this->begin(i) : that.begin(i);
            new_end[i] = this->end(i) <= that.end(i) ? this->end(i) : that.end(i);
        }

        return Bounds::from_bounds(new_begin, new_end);
    }

    KMM_HOST_DEVICE
    bool overlaps(const Bounds& that) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= this->begin(i) < that.end(i);
            result &= that.begin(i) < this->end(i);
            result &= this->begin(i) < this->end(i);
            result &= that.begin(i) < that.end(i);
        }

        return result;
    }

    KMM_HOST_DEVICE
    bool contains(const Bounds& that) const {
        bool contains = true;
        bool is_empty = false;

        for (size_t i = 0; is_less(i, N); i++) {
            contains &= that.begin(i) >= this->begin(i);
            contains &= that.end(i) <= this->end(i);
            is_empty |= that.begin(i) >= that.end(i);
        }

        return contains || is_empty;
    }

    KMM_HOST_DEVICE
    bool contains(const Index<N, T>& that) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= (this)[i].contains(that[i]);
        }

        return result;
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) + 1 == N)>::type>
    KMM_HOST_DEVICE bool contains(const T& first, Ts&&... rest) {
        return contains(Index<N, T> {first, rest...});
    }

    KMM_HOST_DEVICE
    Bounds intersection(const Dim<N, T>& that) const {
        return intersection(Bounds<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool overlaps(const Dim<N, T>& that) const {
        return overlaps(Bounds<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool contains(const Dim<N, T>& that) const {
        return contains(Bounds<N, T> {that});
    }

    KMM_HOST_DEVICE
    Bounds shift_by(const Index<N, T>& offset) const {
        storage_type result = *this;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = (*this)[i].shift_by(offset[i]);
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE
    Bounds split_off_along(size_t axis, const T& mid) const {
        if (is_less(axis, N)) {
            auto result = *this;
            result[axis] = (*this)[axis].split_at(mid);
            return result;
        } else {
            return Bounds::empty();
        }
    }
};

template<typename... Ts>
Index(Ts&&...) -> Index<sizeof...(Ts)>;

template<typename... Ts>
Dim(Ts&&...) -> Dim<sizeof...(Ts)>;

template<typename... Ts>
Bounds(Ts&&...) -> Bounds<sizeof...(Ts)>;

template<typename T>
Range(const T&) -> Range<T>;

template<typename T>
Range(const T&, const T&) -> Range<T>;

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Index<N + M, T> concat(const Index<N, T>& lhs, const Index<M, T>& rhs) {
    return Index<N + M, T> {
        concat((const fixed_array<T, N>&)(lhs), (const fixed_array<T, M>&)(rhs))};
}

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Dim<N + M, T> concat(const Dim<N, T>& lhs, const Dim<M, T>& rhs) {
    return Index<N + M, T> {
        concat((const fixed_array<T, N>&)(lhs), (const fixed_array<T, M>&)(rhs))};
}

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Bounds<N + M, T> concat(const Bounds<N, T>& lhs, const Bounds<M, T>& rhs) {
    return Bounds<N + M, T> {
        concat((const fixed_array<Range<T>, N>&)(lhs), (const fixed_array<Range<T>, M>&)(rhs))};
}

}  // namespace kmm

#include <iosfwd>

namespace kmm {

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Index<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Dim<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Range<T>& p) {
    return stream << p.begin << "..." << p.end;
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Bounds<N, T>& p) {
    stream << "{";
    for (size_t i = 0; is_less(i, N); i++) {
        if (i != 0) {
            stream << ", ";
        }

        stream << p;
    }

    return stream << "}";
}
}  // namespace kmm

#include "fmt/ostream.h"

template<size_t N, typename T>
struct fmt::formatter<kmm::Index<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Dim<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Bounds<N, T>>: fmt::ostream_formatter {};

#include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct std::hash<kmm::Index<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Dim<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Bounds<N, T>> {
    size_t operator()(const kmm::Bounds<N, T>& p) const {
        kmm::fixed_array<T, N> v[2] = {p.begin(), p.end()};
        return kmm::hash_range(v, v + 2);
    }
};