#pragma once

namespace kmm {

namespace details {
template<typename T>
static constexpr bool is_signed_integral = T(-1) < T(0);
}

/**
 * Divide `num` by `denom` and round the result down.
 */
template<typename T>
constexpr T div_floor(T a, T b) {
    constexpr T zero = static_cast<T>(0);
    T quotient = a / b;

    if constexpr (details::is_signed_integral<T>) {
        // Adjust the quotient if a and b have different signs
        if (a % b != zero && ((a >= zero) ^ (b >= zero))) {
            quotient -= 1;
        }
    }

    return quotient;
}

/**
 * Divide `num` by `denom` and round the result up.
 */
template<typename T>
constexpr T div_ceil(T a, T b) {
    constexpr T zero = static_cast<T>(0);
    T quotient = a / b;

    // Adjust the quotient
    if (a % b != zero) {
        if constexpr (details::is_signed_integral<T>) {
            // Adjust the quotient if both a and b have the same sign
            if ((a >= zero) == (b >= zero)) {
                quotient += 1;
            }
        } else {
            quotient += 1;
        }
    }

    return quotient;
}

/**
 * Round `input` to the first multiple of `multiple`.
 *
 * In other words, returns the smallest value not less than `input` that is divisible by `multiple`.
 */
template<typename T>
constexpr T round_up_to_multiple(T input, T multiple) {
    constexpr T zero = static_cast<T>(0);

    if constexpr (details::is_signed_integral<T>) {
        if (multiple < zero) {
            multiple = -multiple;
        }
    }

    T remainder = input % multiple;

    if (remainder == zero) {
        return input;
    } else {
        if constexpr (details::is_signed_integral<T>) {
            if (input < zero) {
                return input - remainder;
            } else {
                return input - remainder + multiple;
            }
        } else {
            return input - remainder + multiple;
        }
    }
}

/**
 * Return the smallest integer that is a power of two and is not less than `input`.
 */
template<typename T>
constexpr T round_up_to_power_of_two(T input) {
    if (input <= static_cast<T>(0)) {
        return static_cast<T>(1);
    }

    input -= static_cast<T>(1);
    for (decltype(sizeof(T)) i = 1; i < sizeof(T) * 8; i *= 2) {
        input |= (input >> i);
    }

    input += static_cast<T>(1);
    return input;
}

/**
 * Check if the given integer is a power of two.
 */
template<typename T>
static bool is_power_of_two(T input) {
    if (input <= static_cast<T>(0)) {
        return false;
    }

    return (input & (input - 1)) == static_cast<T>(0);
}

}  // namespace kmm
