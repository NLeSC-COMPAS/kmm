#pragma once

#include "kmm/utils/macros.hpp"

namespace kmm {

/**
 * Pair of a key (type `int64_t`) and an associated value (type `T`).
 *
 * The main purpose is that `KeyValue` pairs implement the `>` operator, allowed them to be sorted.
 * The pairs are ordered by value, with ties resolved by the key.
 */
template<typename V>
struct alignas(alignof(long) >= alignof(V) ? 2 * alignof(long) : alignof(V)) KeyValue {
    using key_type = long;
    using value_type = V;

    constexpr KeyValue() = default;

    KMM_HOST_DEVICE
    constexpr KeyValue(key_type k, value_type v) : value(v), key(k) {}

    value_type value;
    key_type key;
};

template<typename T>
KMM_HOST_DEVICE bool operator==(const KeyValue<T>& a, const KeyValue<T>& b) {
    return (a.value == b.value) & (a.key == b.key);
}

template<typename T>
KMM_HOST_DEVICE bool operator<(const KeyValue<T>& a, const KeyValue<T>& b) {
    // The obvious expression would be:
    //  `(a.value < b.value` || (a.value == b.value && a.key < b.key)`.
    // However, this would no work well with `NaN`, since `a.value == b.value` would then
    // become false. Instead, we use `!(b.value < a.value)` the check if `a` and `b` have
    // the same value, since this properly gives `true` for `NaN` values.
    return (a.value < b.value) | ((!(b.value < a.value)) & (a.key < b.key));
}

template<typename T>
KMM_HOST_DEVICE bool operator<=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return (a.value < b.value) | ((!(b.value < a.value)) & (a.key <= b.key));
}

template<typename T>
KMM_HOST_DEVICE bool operator!=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return !(a == b);
}

template<typename T>
KMM_HOST_DEVICE bool operator>(const KeyValue<T>& a, const KeyValue<T>& b) {
    return b < a;
}

template<typename T>
KMM_HOST_DEVICE bool operator>=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return b <= a;
}

}  // namespace kmm

#if !KMM_IS_RTC
    #include <iostream>

    #include "fmt/ostream.h"

    #include "kmm/utils/hash_utils.hpp"

namespace kmm {

template<typename T>
std::ostream& operator<<(std::ostream& stream, const KeyValue<T>& p) {
    return stream << "{key=" << p.key << ", value=" << p.value << "}";
}
}  // namespace kmm

template<typename T>
struct fmt::formatter<kmm::KeyValue<T>>: fmt::ostream_formatter {};

template<typename T>
struct std::hash<kmm::KeyValue<T>> {
    size_t operator()(const kmm::KeyValue<T>& p) const {
        return kmm::hash_fields(p.key, p.value);
    }
};
#endif