#pragma once

#include "kmm/utils/checked_compare.hpp"
#include "kmm/utils/dim.hpp"
#include "kmm/utils/fixed_vector.hpp"
#include "kmm/utils/macros.hpp"
#include "kmm/utils/point.hpp"
#include "kmm/utils/range.hpp"

namespace kmm {

template<size_t N, typename T = default_index_type>
class Bounds: public fixed_vector<Range<T>, N> {
  public:
    using storage_type = fixed_vector<Range<T>, N>;

    KMM_HOST_DEVICE
    explicit constexpr Bounds(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    Bounds() {
        for (size_t i = 0; is_less(i, N); i++) {
            (*this)[i] = Range<T>(static_cast<T>(0));
        }
    }

    constexpr Bounds(const Bounds&) = default;
    constexpr Bounds(Bounds&&) noexcept = default;

    Bounds& operator=(const Bounds&) = default;
    Bounds& operator=(Bounds&&) noexcept = default;

    template<size_t M, typename U>
    KMM_HOST_DEVICE constexpr Bounds(const Bounds<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Bounds::from(that);
    }

    template<typename... Ts, typename = typename detail::enable_if<(sizeof...(Ts) + 1 == N)>::type>
    KMM_HOST_DEVICE Bounds(Range<T> first, Ts&&... args) : Bounds() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    KMM_HOST_DEVICE Bounds(const Dim<N, T>& shape) {
        *this = from_offset_size(Point<N, T>::zero(), Dim<N, T>::from(shape));
    }

    KMM_HOST_DEVICE static constexpr Bounds from_bounds(
        const Point<N, T>& begin,
        const Point<N, T>& end
    ) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = {begin[i], end[i]};
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE static constexpr Bounds from_offset_size(
        const Point<N, T>& offset,
        const Dim<N, T>& shape
    ) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = Range<T>(shape[i]).shift_by(offset[i]);
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE static constexpr Bounds empty() {
        return Bounds(Dim<N, T>::zero());
    }

    KMM_HOST_DEVICE static constexpr Bounds one() {
        return Bounds(Dim<N, T>::one());
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Bounds from(const fixed_vector<Range<U>, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? Range<T>::from(that[i]) : Range<T>(static_cast<T>(1));
        }

        return Bounds(result);
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (i < M) {
                result &= (*this)[i].template is_convertible_to<U>();
            } else {
                result &= (*this)[i] == Range<T>(static_cast<T>(1));
            }
        }

        return result;
    }

    KMM_HOST_DEVICE
    Range<T> get_or_default(size_t i, Range<T> default_value = static_cast<T>(1)) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(i < N)) {
                return (*this)[i];
            }
        }

        return default_value;
    }

    KMM_HOST_DEVICE
    T begin(size_t axis) const {
        return get_or_default(axis).begin;
    }

    KMM_HOST_DEVICE
    T end(size_t axis) const {
        return get_or_default(axis).end;
    }

    KMM_HOST_DEVICE
    T size(size_t axis) const {
        return get_or_default(axis).size();
    }

    KMM_HOST_DEVICE
    Point<N, T> begin() const {
        Point<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = (*this)[axis].begin;
        }
        return result;
    }

    KMM_HOST_DEVICE
    Point<N, T> end() const {
        Point<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = (*this)[axis].end;
        }
        return result;
    }

    KMM_HOST_DEVICE
    Dim<N, T> size() const {
        Dim<N, T> result;
        for (size_t axis = 0; is_less(axis, N); axis++) {
            result[axis] = (*this)[axis].size();
        }
        return result;
    }

    KMM_HOST_DEVICE
    bool is_empty() const {
        bool result = false;

        for (size_t i = 0; is_less(i, N); i++) {
            result |= this->begin(i) >= this->end(i);
        }

        return result;
    }

    KMM_HOST_DEVICE
    T volume() const {
        T result = 1;

        for (size_t i = 0; is_less(i, N); i++) {
            result *= this->end(i) - this->begin(i);
        }

        return this->is_empty() ? T {0} : result;
    }

    KMM_HOST_DEVICE
    Bounds intersection(const Bounds& that) const {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i].begin = this->begin(i) >= that.begin(i) ? this->begin(i) : that.begin(i);
            result[i].end = this->end(i) <= that.end(i) ? this->end(i) : that.end(i);
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE
    Bounds unite(const Bounds& that) const {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i].begin = this->begin(i) < that.begin(i) ? this->begin(i) : that.begin(i);
            result[i].end = this->end(i) > that.end(i) ? this->end(i) : that.end(i);
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE
    bool overlaps(const Bounds& that) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= this->begin(i) < this->end(i) && that.begin(i) < that.end(i) &&  //
                this->begin(i) < that.end(i) && that.begin(i) < this->end(i);
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
    bool contains(const Point<N, T>& that) const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            result &= (*this)[i].contains(that[i]);
        }

        return result;
    }

    template<typename... Ts, typename = typename detail::enable_if<(sizeof...(Ts) + 1 == N)>::type>
    KMM_HOST_DEVICE bool contains(const T& first, Ts&&... rest) {
        return contains(Point<N, T> {first, rest...});
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
    Bounds shift_by(const Point<N, T>& offset) const {
        storage_type result = *this;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = (*this)[i].shift_by(offset[i]);
        }

        return Bounds(result);
    }

    KMM_HOST_DEVICE
    Bounds split_tail_along(size_t axis, const T& mid) {
        if (is_less(axis, N)) {
            auto result = *this;
            result[axis] = (*this)[axis].split_tail(mid);
            return result;
        } else {
            return Bounds::empty();
        }
    }
};

template<typename... Ts>
Bounds(Ts&&...) -> Bounds<sizeof...(Ts)>;

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Bounds<N + M, T> concat(const Bounds<N, T>& lhs, const Bounds<M, T>& rhs) {
    return Bounds<N + M, T> {
        concat((const fixed_vector<Range<T>, N>&)(lhs), (const fixed_vector<Range<T>, M>&)(rhs))
    };
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator==(const Bounds<N, T>& lhs, const Bounds<M, U>& rhs) {
    bool result = true;

    for (size_t i = 0; i < N || i < M; i++) {
        result &= is_equal(lhs.get_or_default(i), rhs.get_or_default(i));
    }

    return result;
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator!=(const Bounds<N, T>& lhs, const Bounds<M, U>& rhs) {
    return !(lhs == rhs);
}
}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {
template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Bounds<N, T>& p) {
    return stream << static_cast<const fixed_vector<Range<T>, N>&>(p);
}
}  // namespace kmm

template<size_t N, typename T>
struct fmt::formatter<kmm::Bounds<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::Bounds<N, T>>: std::hash<kmm::fixed_vector<T, N>> {};
#endif