#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kmm/core/macros.hpp"
#include "kmm/utils/fixed_array.hpp"

namespace kmm {

using default_geometry_type = int64_t;

template<size_t N, typename T = default_geometry_type>
class Index: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Index(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Index() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = T {};
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Index(T first, Ts&&... args) : Index() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Index from(const fixed_array<U, M>& that) {
        Index result;

        for (size_t i = 0; i < N && is_less(i, M); i++) {
            result[i] = static_cast<T>(that[i]);
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Index fill(T value) {
        Index result;

        for (size_t i = 0; i < N; i++) {
            result[i] = value;
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Index zero() {
        return fill(static_cast<T>(0));
    }

    KMM_HOST_DEVICE
    static constexpr Index one() {
        return fill(static_cast<T>(1));
    }

    KMM_HOST_DEVICE
    T get(size_t axis) const {
        return KMM_LIKELY(axis < N) ? (*this)[axis] : static_cast<T>(0);
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Index<N + M> concat(const Index<M>& that) const {
        fixed_array<T, N + M> result;

        for (size_t i = 0; i < N; i++) {
            result[i] = (*this)[i];
        }

        for (size_t i = 0; is_less(i, M); i++) {
            result[N + i] = that[i];
        }

        return Index<N + M> {result};
    }
};

template<typename T>
class Index<0, T>: public fixed_array<T, 0> {
  public:
    using storage_type = fixed_array<T, 0>;

    KMM_HOST_DEVICE
    explicit constexpr Index(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Index() {}

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Index from(const fixed_array<U, M>& that) {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Index fill(T value) {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Index zero() {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Index one() {
        return {};
    }

    KMM_HOST_DEVICE
    T get(size_t axis) const {
        return T {};
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Index<M> concat(const Index<M>& that) const {
        return that;
    }
};

template<size_t N, typename T = default_geometry_type>
class Size: public fixed_array<T, N> {
  public:
    using storage_type = fixed_array<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Size(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Size() {
        for (size_t i = 0; i < N; i++) {
            (*this)[i] = static_cast<T>(1);
        }
    }

    template<typename... Ts, typename = typename std::enable_if<(sizeof...(Ts) < N)>::type>
    KMM_HOST_DEVICE Size(T first, Ts&&... args) : Size() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    KMM_HOST_DEVICE
    static constexpr Size from_point(const Index<N, T>& that) {
        return Size {that};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Size from(const fixed_array<U, M>& that) {
        Size result;

        for (size_t i = 0; i < N && is_less(i, M); i++) {
            result[i] = that[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    static constexpr Size zero() {
        return from_point(Index<N, T>::zero());
    }

    KMM_HOST_DEVICE
    static constexpr Size one() {
        return from_point(Index<N, T>::one());
    }

    KMM_HOST_DEVICE
    Index<N, T> to_point() const {
        return Index<N, T>::from(*this);
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        bool is_empty = false;

        for (size_t i = 0; i < N; i++) {
            is_empty |= (*this)[i] <= static_cast<T>(0);
        }

        return is_empty;
    }

    KMM_HOST_DEVICE
    T get(size_t i) const {
        return KMM_LIKELY(i < N) ? (*this)[i] : static_cast<T>(1);
    }

    KMM_HOST_DEVICE
    T volume() const {
        if constexpr (N == 0) {
            return static_cast<T>(1);
        }

        if (is_empty()) {
            return static_cast<T>(0);
        }

        T volume = (*this)[0];

        for (size_t i = 1; i < N; i++) {
            volume *= (*this)[i];
        }

        return volume;
    }

    KMM_HOST_DEVICE
    Size intersection(const Size& that) const {
        Size<N, T> new_sizes;

        for (size_t i = 0; i < N; i++) {
            if (that[i] <= 0 || (*this)[i] <= 0) {
                new_sizes[i] = static_cast<T>(0);
            } else if ((*this)[i] <= that[i]) {
                new_sizes[i] = (*this)[i];
            } else {
                new_sizes[i] = that[i];
            }
        }

        return {new_sizes};
    }

    KMM_HOST_DEVICE
    bool overlaps(const Size& that) const {
        return !this->is_empty() && !that.is_empty();
    }

    KMM_HOST_DEVICE
    bool contains(const Size& that) const {
        return that.is_empty() || intersection(that) == that;
    }

    KMM_HOST_DEVICE
    bool contains(const Index<N, T>& that) const {
        for (size_t i = 0; i < N; i++) {
            if (that[i] < static_cast<T>(0) || that[i] >= (*this)[i]) {
                return false;
            }
        }

        return true;
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Size<N + M> concat(const Size<M>& that) const {
        return Size<N + M>(to_point().concat(that.to_point()));
    }
};

template<typename T>
class Size<0, T>: public fixed_array<T, 0> {
  public:
    using storage_type = fixed_array<T, 0>;

    KMM_HOST_DEVICE
    explicit constexpr Size(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Size() {}

    KMM_HOST_DEVICE
    static constexpr Size from_point(const Index<0, T>& that) {
        return {};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Size from(const fixed_array<U, M>& that) {
        return that;
    }

    KMM_HOST_DEVICE
    static constexpr Size zero() {
        return {};
    }

    KMM_HOST_DEVICE
    static constexpr Size one() {
        return {};
    }

    KMM_HOST_DEVICE
    Index<0, T> to_point() const {
        return {};
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        return false;
    }

    KMM_HOST_DEVICE
    T get(size_t i) const {
        return static_cast<T>(1);
    }

    KMM_HOST_DEVICE
    T volume() const {
        return static_cast<T>(1);
    }

    KMM_HOST_DEVICE
    Size intersection(const Size& that) const {
        return {};
    }

    KMM_HOST_DEVICE
    bool overlaps(const Size& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    bool contains(const Size& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    bool contains(const Index<0, T>& that) const {
        return true;
    }

    KMM_HOST_DEVICE
    T operator()(size_t axis = 0) const {
        return get(axis);
    }

    template<size_t M>
    KMM_HOST_DEVICE Size<M> concat(const Size<M>& that) const {
        return that;
    }
};

template<size_t N, typename T = default_geometry_type>
class Bounds {
  public:
    Index<N, T> begin;
    Index<N, T> end;

    Bounds() = default;

    KMM_HOST_DEVICE
    Bounds(Index<N, T> begin, Index<N, T> end) : begin(begin), end(end) {}

    KMM_HOST_DEVICE
    Bounds(Size<N, T> sizes) {
        this->end = sizes.to_point();
    }

    KMM_HOST_DEVICE static constexpr Bounds from_offset_size(Index<N, T> offset, Size<N, T> size) {
        return {offset, offset + size.to_point()};
    }

    KMM_HOST_DEVICE static constexpr Bounds from_point(Index<N, T> offset) {
        return from_offset_size(index, Size<N, T>::one());
    }

    KMM_HOST_DEVICE static constexpr Bounds from_bounds(Index<N, T> begin, Index<N, T> end) {
        return {begin, end};
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE static constexpr Bounds from(const Bounds<M, U>& that) {
        return from_bounds(Index<N, T>(that.begin()), Index<N, T>(that.end()));
    }

    KMM_HOST_DEVICE
    T offset(size_t axis) const {
        return begin[axis];
    }

    KMM_HOST_DEVICE
    Index<N, T> offset() const {
        return begin;
    }

    KMM_HOST_DEVICE
    T size(size_t axis) const {
        return begin[axis] <= end[axis] ? end[axis] - begin[axis] : 0;
    }

    KMM_HOST_DEVICE
    Size<N, T> sizes() const {
        Size<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = size(axis);
        }
        return result;
    }

    KMM_HOST_DEVICE
    T size() const {
        T result = 1;

        for (size_t i = 0; is_less(i, N); i++) {
            result *= begin[i] >= end[i] ? 0 : end[i] - begin[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= begin[i] >= end[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    Bounds intersection(const Bounds& that) const {
        Index<N, T> new_begin;
        Index<N, T> new_end;

        for (size_t i = 0; is_less(i, N); i++) {
            new_begin[i] = this->begin[i] >= that.begin[i] ? this->begin[i] : that.begin[i];
            new_end[i] = this->end[i] <= that.end[i] ? this->end[i] : that.end[i];
        }

        return Bounds::from_bounds(new_begin, new_end);
    }

    KMM_HOST_DEVICE
    bool overlaps(const Bounds& that) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= this->begin[i] < that.end[i];
            result &= that.begin[i] < this->end[i];
            result &= this->begin[i] < this->end[i];
            result &= that.begin[i] < that.end[i];
        }

        return result;
    }

    KMM_HOST_DEVICE
    bool contains(const Bounds& that) const {
        bool contains = true;
        bool is_empty = false;

        for (size_t i = 0; is_less(i, N); i++) {
            contains &= that.begin[i] >= this->begin[i];
            contains &= that.end[i] <= this->end[i];
            is_empty |= that.begin[i] >= that.end[i];
        }

        return contains || is_empty;
    }

    KMM_HOST_DEVICE
    bool contains(const Index<N, T>& that) const {
        bool contains = true;

        for (size_t i = 0; is_less(i, N); i++) {
            contains &= that[i] >= this->begin[i];
            contains &= that[i] < this->end[i];
        }

        return contains;
    }

    KMM_HOST_DEVICE
    Bounds intersection(const Size<N, T>& that) const {
        return intersection(Bounds<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool overlaps(const Size<N, T>& that) const {
        return overlaps(Bounds<N, T> {that});
    }

    KMM_HOST_DEVICE
    bool contains(const Size<N, T>& that) const {
        return contains(Bounds<N, T> {that});
    }

    template<size_t M>
    KMM_HOST_DEVICE Bounds<N + M> concat(const Bounds<M>& that) const {
        return Bounds<N + M>::from_bounds(begin.concat(that.begin), end.concat(that.end));
    }
};

template<typename... Ts>
KMM_HOST_DEVICE_NOINLINE Index(Ts...) -> Index<sizeof...(Ts)>;

template<typename... Ts>
KMM_HOST_DEVICE_NOINLINE Size(Ts...) -> Size<sizeof...(Ts)>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE Bounds(Index<N, T> offset, Size<N, T> sizes) -> Bounds<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE_NOINLINE Bounds(Size<N, T> sizes) -> Bounds<N, T>;

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Index<N, T>& a, const Index<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Index<N, T>& a, const Index<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Size<N, T>& a, const Size<N, T>& b) {
    return (const fixed_array<T, N>&)a == (const fixed_array<T, N>&)b;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Size<N, T>& a, const Size<N, T>& b) {
    return !(a == b);
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator==(const Bounds<N, T>& a, const Bounds<N, T>& b) {
    return a.begin == b.begin && a.end == b.end;
}

template<size_t N, typename T>
KMM_HOST_DEVICE bool operator!=(const Bounds<N, T>& a, const Bounds<N, T>& b) {
    return !(a == b);
}

#define KMM_POINT_OPERATOR_IMPL(OP)                                                       \
    template<size_t N, typename T>                                                        \
    KMM_HOST_DEVICE Index<N, T> operator OP(const Index<N, T>& a, const Index<N, T>& b) { \
        Index<N, T> result;                                                               \
        for (size_t i = 0; is_less(i, N); i++) {                                          \
            result[i] = a[i] OP b[i];                                                     \
        }                                                                                 \
        return result;                                                                    \
    }

KMM_POINT_OPERATOR_IMPL(+);
KMM_POINT_OPERATOR_IMPL(-);
KMM_POINT_OPERATOR_IMPL(*);
KMM_POINT_OPERATOR_IMPL(/);

}  // namespace kmm

#include <iosfwd>

namespace kmm {

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Index<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Size<N, T>& p) {
    return stream << fixed_array<T, N>(p);
}

template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Bounds<N, T>& p) {
    stream << "{";
    for (size_t i = 0; is_less(i, N); i++) {
        if (i != 0) {
            stream << ", ";
        }

        stream << p.begin[i] << "..." << p.end[i];
    }

    return stream << "}";
}
}  // namespace kmm

#include "fmt/ostream.h"

template<size_t N, typename T>
struct fmt::formatter<kmm::Index<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Size<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct fmt::formatter<kmm::Bounds<N, T>>: fmt::ostream_formatter {};

#include "kmm/utils/hash_utils.hpp"

template<size_t N, typename T>
struct std::hash<kmm::Index<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Size<N, T>>: std::hash<kmm::fixed_array<T, N>> {};

template<size_t N, typename T>
struct std::hash<kmm::Bounds<N, T>> {
    size_t operator()(const kmm::Bounds<N, T>& p) const {
        kmm::fixed_array<T, N> v[2] = {p.begin, p.end};
        return kmm::hash_range(v, v + 2);
    }
};
