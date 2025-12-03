#pragma once

#include "checked_compare.hpp"
#include "macros.hpp"

namespace kmm {

namespace detail {

template<typename T>
struct checked_arithmetic_impl;

#define KMM_IMPL_CHECKED_ARITHMETIC(T, ADD_FUN, SUB_FUN, MUL_FUN) \
    template<>                                                    \
    struct checked_arithmetic_impl<T> {                           \
        static bool add(T lhs, T rhs, T* result) {                \
            return ADD_FUN(lhs, rhs, result) == false;            \
        }                                                         \
                                                                  \
        static bool sub(T lhs, T rhs, T* result) {                \
            return SUB_FUN(lhs, rhs, result) == false;            \
        }                                                         \
                                                                  \
        static bool mul(T lhs, T rhs, T* result) {                \
            return MUL_FUN(lhs, rhs, result) == false;            \
        }                                                         \
    };

KMM_IMPL_CHECKED_ARITHMETIC(
    signed int,
    __builtin_sadd_overflow,
    __builtin_ssub_overflow,
    __builtin_smul_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    signed long,
    __builtin_saddl_overflow,
    __builtin_ssubl_overflow,
    __builtin_smull_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    signed long long,
    __builtin_saddll_overflow,
    __builtin_ssubll_overflow,
    __builtin_smulll_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    unsigned int,
    __builtin_uadd_overflow,
    __builtin_usub_overflow,
    __builtin_umul_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    unsigned long,
    __builtin_uaddl_overflow,
    __builtin_usubl_overflow,
    __builtin_umull_overflow
)

KMM_IMPL_CHECKED_ARITHMETIC(
    unsigned long long,
    __builtin_uaddll_overflow,
    __builtin_usubll_overflow,
    __builtin_umulll_overflow
)

#define KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(T, R)                   \
    template<>                                                      \
    struct checked_arithmetic_impl<T> {                             \
        static bool add(T lhs, T rhs, T* result) {                  \
            R temp = static_cast<R>(lhs) + static_cast<R>(rhs);     \
            *result = static_cast<T>(temp);                         \
            return detail::checked_convert_impl<R, T>::check(temp); \
        }                                                           \
                                                                    \
        static bool sub(T lhs, T rhs, T* result) {                  \
            R temp = static_cast<R>(lhs) - static_cast<R>(rhs);     \
            *result = static_cast<T>(temp);                         \
            return detail::checked_convert_impl<R, T>::check(temp); \
        }                                                           \
                                                                    \
        static bool mul(T lhs, T rhs, T* result) {                  \
            R temp = static_cast<R>(lhs) * static_cast<R>(rhs);     \
            *result = static_cast<T>(temp);                         \
            return detail::checked_convert_impl<R, T>::check(temp); \
        }                                                           \
    };

KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(signed short, signed int)
KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(unsigned short, signed int)

KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(signed char, signed int)
KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(unsigned char, signed int)
KMM_IMPL_CHECKED_ARITHMETIC_FORWARD(char, signed int)

}  // namespace detail

template<typename T>
KMM_HOST_DEVICE T checked_add(const T& left, const T& right) {
    T output;

    if (!detail::checked_arithmetic_impl<T>::add(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<typename T>
KMM_HOST_DEVICE T checked_sub(const T& left, const T& right) {
    T output;

    if (!detail::checked_arithmetic_impl<T>::sub(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<typename T>
KMM_HOST_DEVICE T checked_mul(const T& left, const T& right) {
    T output;

    if (!detail::checked_arithmetic_impl<T>::mul(left, right, &output)) {
        throw_overflow_exception();
    }

    return output;
}

template<typename T>
KMM_HOST_DEVICE T checked_neg(const T& input) {
    return checked_sub(static_cast<T>(0), input);
}

template<typename T, typename U = T>
KMM_HOST_DEVICE U checked_sum(const T* begin, const T* end, U initial = T(0)) {
    bool is_valid = true;
    U accum = initial;

    for (const T* it = begin; it != end; it++) {
        is_valid &= is_convertible<U>(*it);
        is_valid &= detail::checked_arithmetic_impl<U>::add(static_cast<U>(*it), accum, &accum);
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return accum;
}

template<typename T, typename U = T>
KMM_HOST_DEVICE U checked_product(const T* begin, const T* end, U initial = T(1)) {
    bool is_valid = true;
    U accum = initial;

    for (const T* it = begin; it != end; it++) {
        is_valid &= is_convertible<U>(*it);
        is_valid &= detail::checked_arithmetic_impl<U>::mul(static_cast<U>(*it), accum, &accum);
    }

    if (!is_valid) {
        throw_overflow_exception();
    }

    return accum;
}

}  // namespace kmm