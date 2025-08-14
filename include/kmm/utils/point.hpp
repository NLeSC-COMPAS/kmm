#pragma once

#include "kmm/utils/checked_compare.hpp"
#include "kmm/utils/fixed_vector.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

using default_index_type = signed long int;  // int64_t

namespace detail {
template<bool, typename = void>
struct enable_if {};

template<typename T>
struct enable_if<true, T> {
    using type = T;
};
}  // namespace detail

template<size_t N, typename T = default_index_type>
class Point: public fixed_vector<T, N> {
  public:
    using storage_type = fixed_vector<T, N>;

    KMM_HOST_DEVICE
    explicit constexpr Point(const storage_type& storage) : storage_type(storage) {}

    KMM_HOST_DEVICE
    constexpr Point() {
        for (size_t i = 0; is_less(i, N); i++) {
            (*this)[i] = T {};
        }
    }

    template<typename... Ts, typename = typename detail::enable_if<(sizeof...(Ts) + 1 == N)>::type>
    KMM_HOST_DEVICE Point(T first, Ts&&... args) : Point() {
        (*this)[0] = first;

        size_t index = 0;
        (((*this)[++index] = args), ...);
    }

    template<size_t M, typename U>
    KMM_HOST_DEVICE constexpr Point(const Point<M, U>& that) {
        if (!that.template is_convertible_to<N, T>()) {
            throw_overflow_exception();
        }

        *this = Point::from(that);
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE static constexpr Point from(const fixed_vector<U, M>& that) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = is_less(i, M) ? static_cast<T>(that[i]) : static_cast<T>(0);
        }

        return Point(result);
    }

    KMM_HOST_DEVICE
    static Point fill(T value) {
        storage_type result;

        for (size_t i = 0; is_less(i, N); i++) {
            result[i] = value;
        }

        return Point(result);
    }

    KMM_HOST_DEVICE
    static Point one() {
        return fill(static_cast<T>(1));
    }

    KMM_HOST_DEVICE
    static Point zero() {
        return fill(static_cast<T>(0));
    }

    template<size_t M = N, typename U = T>
    KMM_HOST_DEVICE bool is_convertible_to() const {
        bool result = true;

        for (size_t i = 0; is_less(i, N); i++) {
            if (is_less(i, M)) {
                result &= is_convertible<U>((*this)[i]);
            } else {
                result &= is_equal((*this)[i], static_cast<T>(0));
            }
        }

        return result;
    }

    KMM_HOST_DEVICE
    T get_or_default(size_t i, T default_value = {}) const {
        if constexpr (N > 0) {
            if (KMM_LIKELY(is_less(i, N))) {
                return (*this)[i];
            }
        }

        return default_value;
    }
};

template<typename... Ts>
Point(Ts&&...) -> Point<sizeof...(Ts)>;

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE Point<N + M, T> concat(const Point<N, T>& lhs, const Point<M, T>& rhs) {
    return Point<N + M, T> {
        concat((const fixed_vector<T, N>&)(lhs), (const fixed_vector<T, M>&)(rhs))
    };
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator==(const Point<N, T>& lhs, const Point<M, U>& rhs) {
    bool result = true;

    for (size_t i = 0; is_less(i, N) || is_less(i, M); i++) {
        result &= is_equal(lhs.get_or_default(i), rhs.get_or_default(i));
    }

    return result;
}

template<size_t N, typename T, size_t M, typename U>
KMM_HOST_DEVICE bool operator!=(const Point<N, T>& lhs, const Point<M, U>& rhs) {
    return !(lhs == rhs);
}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iosfwd>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {
template<size_t N, typename T>
std::ostream& operator<<(std::ostream& stream, const Point<N, T>& p) {
    return stream << static_cast<const fixed_vector<T, N>&>(p);
}
}  // namespace kmm

template<size_t N, typename T>
struct fmt::formatter<kmm::Point<N, T>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::Point<N, T>>: std::hash<kmm::fixed_vector<T, N>> {};
#endif