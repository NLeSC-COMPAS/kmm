#pragma once

#include "checked_compare.hpp"
#include "panic.hpp"

namespace kmm {

#define KMM_PANIC_OUT_OF_BOUNDS() KMM_PANIC("access out of bounds")

template<typename T>
static constexpr size_t compute_fixed_vector_alignment(size_t num) {
    constexpr size_t max_align = 16;
    size_t align = alignof(T);
    while (align < max_align && align < num * sizeof(T)) {
        align *= 2;
    }
    return align;
}

template<typename T, size_t N>
struct alignas(compute_fixed_vector_alignment<T>(N)) fixed_vector {
    KMM_HOST_DEVICE
    T& operator[](size_t i) {
        if (i >= N) {
            KMM_PANIC_OUT_OF_BOUNDS();
        }

        return __internal_data[i];
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t i) const {
        if (i >= N) {
            KMM_PANIC_OUT_OF_BOUNDS();
        }

        return __internal_data[i];
    }

    T __internal_data[N] {};
};

template<typename T>
struct fixed_vector<T, 0> {
    KMM_HOST_DEVICE
    T& operator[](size_t i) {
        KMM_PANIC_OUT_OF_BOUNDS();
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t i) const {
        KMM_PANIC_OUT_OF_BOUNDS();
    }
};

template<typename T>
struct alignas(compute_fixed_vector_alignment<T>(1)) fixed_vector<T, 1> {
    KMM_HOST_DEVICE
    T& operator[](size_t i) {
        switch (i) {
            case 0:
                return x;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t i) const {
        switch (i) {
            case 0:
                return x;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    KMM_HOST_DEVICE
    operator T() const {
        return x;
    }

    KMM_HOST_DEVICE
    fixed_vector& operator=(T value) {
        x = value;
        return *this;
    }

    T x {};
};

template<typename T>
struct alignas(compute_fixed_vector_alignment<T>(2)) fixed_vector<T, 2> {
    KMM_HOST_DEVICE
    T& operator[](size_t i) {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t i) const {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    T x {};
    T y {};
};

template<typename T>
struct alignas(compute_fixed_vector_alignment<T>(3)) fixed_vector<T, 3> {
    KMM_HOST_DEVICE
    T& operator[](size_t i) {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t i) const {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    T x {};
    T y {};
    T z {};
};

template<typename T>
struct alignas(compute_fixed_vector_alignment<T>(4)) fixed_vector<T, 4> {
    KMM_HOST_DEVICE
    T& operator[](size_t i) {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    KMM_HOST_DEVICE
    const T& operator[](size_t i) const {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                KMM_PANIC_OUT_OF_BOUNDS();
        }
    }

    T x {};
    T y {};
    T z {};
    T w {};
};

template<typename T, typename U, size_t N, size_t M>
KMM_HOST_DEVICE bool operator==(const fixed_vector<T, N>& lhs, const fixed_vector<U, M>& rhs) {
    if (N != M) {
        return false;
    }

    bool result = true;

    for (size_t i = 0; is_less(i, N); i++) {
        result &= is_equal(lhs[i], rhs[i]);
    }

    return result;
}

template<typename T, typename U, size_t N, size_t M>
KMM_HOST_DEVICE bool operator!=(const fixed_vector<T, N>& lhs, const fixed_vector<U, M>& rhs) {
    return !(lhs == rhs);
}

template<typename T, size_t N, size_t M>
KMM_HOST_DEVICE fixed_vector<T, N + M> concat(
    const fixed_vector<T, N>& lhs,
    const fixed_vector<T, M>& rhs
) {
    fixed_vector<T, N + M> result;

    for (size_t i = 0; is_less(i, N); i++) {
        result[i] = lhs[i];
    }

    for (size_t i = 0; is_less(i, M); i++) {
        result[i + N] = rhs[i];
    }

    return result;
}
}  // namespace kmm

#if !KMM_IS_RTC
    #include <iostream>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {

template<typename T, size_t N>
std::ostream& operator<<(std::ostream& stream, const fixed_vector<T, N>& p) {
    stream << "{";
    for (size_t i = 0; is_less(i, N); i++) {
        if (i != 0) {
            stream << ", ";
        }

        stream << p[i];
    }

    return stream << "}";
}
}  // namespace kmm

template<typename T, size_t N>
struct fmt::formatter<kmm::fixed_vector<T, N>>: fmt::ostream_formatter {};

template<size_t N, typename T>
struct std::hash<kmm::fixed_vector<T, N>> {
    size_t operator()(const kmm::fixed_vector<T, N>& p) const {
        size_t result = 0;
        for (size_t i = 0; i < N; i++) {
            kmm::hash_combine(result, p[i]);
        }
        return result;
    }
};

template<typename T>
struct std::hash<kmm::fixed_vector<T, 0>> {
    size_t operator()(const kmm::fixed_vector<T, 0>& p) const {
        return 0;
    }
};
#endif