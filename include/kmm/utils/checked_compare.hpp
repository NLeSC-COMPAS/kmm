#pragma once

#include <cstdio>
#include <cstdlib>

#include "kmm/utils/macros.hpp"
#include "kmm/utils/panic.hpp"

namespace kmm {

namespace detail {

enum class numeric_type_tag {  //
    signed_int,
    unsigned_int,
    floating_point,
    other
};

template<typename T>
struct numeric_type_traits {
    static constexpr numeric_type_tag tag = numeric_type_tag::other;
};

template<>
struct numeric_type_traits<float> {
    static constexpr numeric_type_tag tag = numeric_type_tag::floating_point;
};

template<>
struct numeric_type_traits<double> {
    static constexpr numeric_type_tag tag = numeric_type_tag::floating_point;
};

#define KMM_DEFINE_INT_TRAITS(T)                                                          \
    template<>                                                                            \
    struct numeric_type_traits<signed T> {                                                \
        static constexpr numeric_type_tag tag = numeric_type_tag::signed_int;             \
        using unsigned_type = unsigned T;                                                 \
                                                                                          \
        static constexpr signed T max_inclusive =                                         \
            (signed T)(unsigned_type(~unsigned_type(0)) >> 1);                            \
        static constexpr signed T min_inclusive = ~max_inclusive;                         \
                                                                                          \
        static constexpr float min_inclusive_float = min_inclusive;                       \
        static constexpr float max_exclusive_float = unsigned_type(max_inclusive) + 1;    \
    };                                                                                    \
                                                                                          \
    template<>                                                                            \
    struct numeric_type_traits<unsigned T> {                                              \
        static constexpr numeric_type_tag tag = numeric_type_tag::unsigned_int;           \
                                                                                          \
        static constexpr unsigned T min_inclusive = 0;                                    \
        static constexpr unsigned T max_inclusive =                                       \
            static_cast<unsigned T>(~static_cast<unsigned T>(0));                         \
                                                                                          \
        static constexpr float min_inclusive_float = min_inclusive;                       \
        static constexpr float max_exclusive_float = 2.0f * float(max_inclusive / 2 + 1); \
    };

KMM_DEFINE_INT_TRAITS(char)
KMM_DEFINE_INT_TRAITS(short)
KMM_DEFINE_INT_TRAITS(int)
KMM_DEFINE_INT_TRAITS(long)
KMM_DEFINE_INT_TRAITS(long long)

template<>
struct numeric_type_traits<char> {
    static constexpr bool is_signed = static_cast<char>(-1) < 0;
    using unsigned_type = unsigned char;

    static constexpr numeric_type_tag tag = is_signed  //
        ? numeric_type_tag::signed_int
        : numeric_type_tag::unsigned_int;

    static constexpr char max_inclusive = is_signed  //
        ? numeric_type_traits<signed char>::max_inclusive
        : numeric_type_traits<unsigned char>::min_inclusive;

    static constexpr char min_inclusive = is_signed  //
        ? numeric_type_traits<signed char>::min_inclusive
        : numeric_type_traits<unsigned char>::min_inclusive;

    static constexpr float min_inclusive_float = float(min_inclusive);
    static constexpr float max_exclusive_float = float(max_inclusive) + 1.0F;
};

template<
    typename L,
    typename R,
    numeric_type_tag = numeric_type_traits<L>::tag,
    numeric_type_tag = numeric_type_traits<R>::tag>
struct checked_compare_impl;

template<typename T>
struct checked_compare_impl<T, T, numeric_type_tag::other, numeric_type_tag::other> {
    KMM_HOST_DEVICE
    static constexpr bool is_equal(const T& left, const T& right) {
        return left == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(const T& left, const T& right) {
        return left < right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::signed_int, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return left == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return left < right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::unsigned_int, numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return left == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return left < right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::unsigned_int, numeric_type_tag::signed_int> {
    using UR = typename numeric_type_traits<R>::unsigned_type;

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return right >= static_cast<R>(0) && left == static_cast<UR>(right);
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return right >= static_cast<R>(0) && left < static_cast<UR>(right);
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::signed_int, numeric_type_tag::unsigned_int> {
    using UL = typename numeric_type_traits<L>::unsigned_type;

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return left >= static_cast<L>(0) && static_cast<UL>(left) == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return left < static_cast<L>(0) || static_cast<UL>(left) < right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<
    L,
    R,
    numeric_type_tag::floating_point,
    numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return left == right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        return left < right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::floating_point, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (floor(left) < numeric_type_traits<R>::min_inclusive_float) {
            return true;
        }

        if (floor(left) >= numeric_type_traits<R>::max_exclusive_float) {
            return false;
        }

        return static_cast<R>(floor(left)) < right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        if (floor(left) != left) {
            return false;
        }

        if (left < numeric_type_traits<R>::min_inclusive_float) {
            return false;
        }

        if (left >= numeric_type_traits<R>::max_exclusive_float) {
            return false;
        }

        return static_cast<R>(left) == right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<L, R, numeric_type_tag::signed_int, numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (ceil(right) < numeric_type_traits<L>::min_inclusive_float) {
            return false;
        }

        if (ceil(right) >= numeric_type_traits<L>::max_exclusive_float) {
            return true;
        }

        return left < static_cast<L>(ceil(right));
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return checked_compare_impl<R, L>::is_equal(right, left);
    }
};

template<typename L, typename R>
struct checked_compare_impl<
    L,
    R,
    numeric_type_tag::floating_point,
    numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (floor(left) < numeric_type_traits<R>::min_inclusive_float) {
            return true;
        }

        if (floor(left) >= numeric_type_traits<R>::max_exclusive_float) {
            return false;
        }

        return static_cast<R>(floor(left)) < right;
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        if (floor(left) != left) {
            return false;
        }

        if (left < numeric_type_traits<R>::min_inclusive_float) {
            return false;
        }

        if (left >= numeric_type_traits<R>::max_exclusive_float) {
            return false;
        }

        return static_cast<R>(left) == right;
    }
};

template<typename L, typename R>
struct checked_compare_impl<
    L,
    R,
    numeric_type_tag::unsigned_int,
    numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static constexpr bool is_less(L left, R right) {
        if (ceil(right) < numeric_type_traits<L>::min_inclusive_float) {
            return false;
        }

        if (ceil(right) >= numeric_type_traits<L>::max_exclusive_float) {
            return true;
        }

        return left < static_cast<L>(ceil(right));
    }

    KMM_HOST_DEVICE
    static constexpr bool is_equal(L left, R right) {
        return checked_compare_impl<R, L>::is_equal(right, left);
    }
};

template<
    typename I,
    typename O,
    numeric_type_tag = numeric_type_traits<I>::tag,
    numeric_type_tag = numeric_type_traits<O>::tag>
struct checked_convert_impl;

template<typename T>
struct checked_convert_impl<T, T, numeric_type_tag::other, numeric_type_tag::other> {
    static bool check(const T& input) {
        return true;
    }
};

template<typename I, typename O>
struct checked_convert_impl<I, O, numeric_type_tag::signed_int, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        return static_cast<O>(input) == input;
    }
};

template<typename I, typename O>
struct checked_convert_impl<I, O, numeric_type_tag::unsigned_int, numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        return static_cast<O>(input) == input;
    }
};

template<typename I, typename O>
struct checked_convert_impl<
    I,
    O,
    numeric_type_tag::floating_point,
    numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        return (input != input) || input == static_cast<O>(input);
    }
};

template<typename I, typename O>
struct checked_convert_impl<I, O, numeric_type_tag::unsigned_int, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        return !checked_compare_impl<O, I>::is_less(numeric_type_traits<O>::max_inclusive, input);
    }
};

template<typename I, typename O>
struct checked_convert_impl<I, O, numeric_type_tag::signed_int, numeric_type_tag::unsigned_int> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        return input >= static_cast<I>(0) &&  //
            !checked_compare_impl<O, I>::is_less(numeric_type_traits<O>::max_inclusive, input);
    }
};

template<typename I, typename O>
struct checked_convert_impl<I, O, numeric_type_tag::floating_point, numeric_type_tag::signed_int> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        if (trunc(input) != input) {
            return false;
        }

        if (input < numeric_type_traits<O>::min_inclusive_float) {
            return false;
        }

        if (input >= numeric_type_traits<O>::max_exclusive_float) {
            return false;
        }

        return true;
    }
};

template<typename I, typename O>
struct checked_convert_impl<
    I,
    O,
    numeric_type_tag::floating_point,
    numeric_type_tag::unsigned_int> {
    static bool check(const I& input) {
        if (trunc(input) != input) {
            return false;
        }

        if (input < static_cast<I>(0)) {
            return false;
        }

        if (input >= numeric_type_traits<O>::max_exclusive_float) {
            return false;
        }

        return true;
    }
};

template<typename I, typename O>
struct checked_convert_impl<I, O, numeric_type_tag::signed_int, numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        return checked_compare_impl<I, O>::is_equal(input, static_cast<O>(input));
    }
};

template<typename I, typename O>
struct checked_convert_impl<
    I,
    O,
    numeric_type_tag::unsigned_int,
    numeric_type_tag::floating_point> {
    KMM_HOST_DEVICE
    static bool check(const I& input) {
        return checked_compare_impl<I, O>::is_equal(input, static_cast<O>(input));
    }
};

}  // namespace detail

#if KMM_IS_DEVICE
// on the GPU, we just panic immediately
KMM_DEVICE void throw_overflow_exception() {
    KMM_PANIC("overflow occurred in operation");
}
#else
// on the host, we can throw an exception
[[noreturn]] void throw_overflow_exception();
#endif

template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_less(const L& left, const R& right) {
    return detail::checked_compare_impl<L, R>::is_less(left, right);
}

template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_equal(const L& left, const R& right) {
    return detail::checked_compare_impl<L, R>::is_equal(left, right);
}

template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_less_equal(const L& left, const R& right) {
    return is_less(left, right) || is_equal(left, right);
}

template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_greater(const L& left, const R& right) {
    return is_less(right, left);
}

template<typename L, typename R>
KMM_HOST_DEVICE constexpr bool is_greater_equal(const L& left, const R& right) {
    return is_less_equal(right, left);
}

template<typename U, typename T>
KMM_HOST_DEVICE bool is_convertible(const T& input) {
    return detail::checked_convert_impl<T, U>::check(input);
}

template<typename U, typename T>
KMM_HOST_DEVICE constexpr bool in_range(const T& input, const U& length) {
    return !is_less(input, 0) && is_less(input, length) && is_convertible<U>(input);
}

template<typename U, typename T>
KMM_HOST_DEVICE constexpr U checked_cast(const T& input) {
    if (!is_convertible<U>(input)) {
        throw_overflow_exception();
    }

    return static_cast<U>(input);
}

}  // namespace kmm