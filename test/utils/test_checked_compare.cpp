#include <cstdio>
#include <cstdlib>

#include "catch2/catch_all.hpp"

#include "kmm/utils/checked_compare.hpp"

using namespace kmm;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

#define REQUIRE_IS_LESS(A, B)     \
    REQUIRE_FALSE(is_less(A, A)); \
    REQUIRE_FALSE(is_less(B, B)); \
    REQUIRE(is_less(A, B));       \
    REQUIRE_FALSE(is_less(B, A));

TEST_CASE("is_less") {
    REQUIRE_IS_LESS(i8(-128), u8(0));
    REQUIRE_IS_LESS(u8(0), i8(127));
    REQUIRE_IS_LESS(i16(-1), u16(1));
    REQUIRE_IS_LESS(u16(1), i16(2));
    REQUIRE_IS_LESS(i32(-100), u32(0));
    REQUIRE_IS_LESS(u32(0), i64(1));
    REQUIRE_IS_LESS(i64(std::numeric_limits<i64>::min()), u64(0ULL));
    REQUIRE_IS_LESS(u64(0ULL), i64(std::numeric_limits<i64>::max()));
    REQUIRE_IS_LESS(f32(1.5f), f64(1.6));
    REQUIRE_IS_LESS(f64(-1.0), f32(0.0f));
    REQUIRE_IS_LESS(f32(-std::numeric_limits<f32>::max()), f64(std::numeric_limits<f64>::max()));
    REQUIRE_IS_LESS(i32(std::numeric_limits<i32>::min()), i32(0));
    REQUIRE_IS_LESS(i32(-2147483647), u32(2147483648u));
    REQUIRE_IS_LESS(u16(0), f64(0.1));
    REQUIRE_IS_LESS(f64(-std::numeric_limits<f64>::infinity()), f64(-1e308));
    REQUIRE_IS_LESS(f32(-1e10f), f64(-1e9));
    REQUIRE_IS_LESS(u64(100), f64(101.0));
    REQUIRE_IS_LESS(i64(-1000), f32(0.0f));
    REQUIRE_IS_LESS(f32(0.999f), i32(1));
    REQUIRE_IS_LESS(i8(-10), f32(-9.9f));
    REQUIRE_IS_LESS(u8(200), u16(300));
    REQUIRE_IS_LESS(i16(30000), i32(40000));
    REQUIRE_IS_LESS(i32(1), f64(1.1));
    REQUIRE_IS_LESS(f64(1.1), f64(1.2L));
    REQUIRE_IS_LESS(f64(-1.2L), f64(-1.1));
    REQUIRE_IS_LESS(f64(0.0L), f64(0.1L));
    REQUIRE_IS_LESS(i64(-1), f64(0.0));
    REQUIRE_IS_LESS(u32(1), f64(1.1));
    REQUIRE_IS_LESS(f64(-0.01), u32(0));
    REQUIRE_IS_LESS(f32(-0.1f), u16(0));
    REQUIRE_IS_LESS(u64(1), f64(1.0001L));
    REQUIRE_IS_LESS(i8(10), f64(10.1L));
    REQUIRE_IS_LESS(i16(123), f64(123.1L));
    REQUIRE_IS_LESS(u16(65535), f64(65535.5));
    REQUIRE_IS_LESS(i32(2147483646), i64(2147483647));
    REQUIRE_IS_LESS(u32(4294967294U), u64(4294967295ULL));
    REQUIRE_IS_LESS(i64(9223372036854775806LL), u64(9223372036854775807ULL));
    REQUIRE_IS_LESS(f64(0.0), std::numeric_limits<f64>::infinity());
    REQUIRE_IS_LESS(f32(-std::numeric_limits<f32>::infinity()), f32(0.0f));
    REQUIRE_IS_LESS(i32(-1), f64(0.0L));
    REQUIRE_IS_LESS(f32(std::numeric_limits<f32>::max()), f64(std::numeric_limits<f64>::max()));
    REQUIRE_IS_LESS(i32(-1), u32(10));
    REQUIRE_IS_LESS(f64(0.0), std::numeric_limits<f64>::infinity());
    REQUIRE_IS_LESS(i64(0), u64(std::numeric_limits<u64>::max()));
    REQUIRE_IS_LESS(f32(1.0001f), f64(1.0002));
    REQUIRE_IS_LESS(i32(-1), i32(0));

    REQUIRE_FALSE(is_less(-0.0, 0.0));
    REQUIRE_FALSE(is_less(f32(1.1f), f64(1.1)));
    REQUIRE_FALSE(is_less(u8(255), u8(255)));
    REQUIRE_FALSE(is_less(std::numeric_limits<f64>::quiet_NaN(), f64(0.0)));
}

#define REQUIRE_IS_EQUAL(A, B)    \
    REQUIRE(is_equal(A, B));      \
    REQUIRE_FALSE(is_less(A, B)); \
    REQUIRE_FALSE(is_less(B, A));

#define REQUIRE_NOT_EQUAL(A, B)    \
    REQUIRE_FALSE(is_equal(A, B)); \
    REQUIRE((is_less(A, B) || is_less(B, A)));

TEST_CASE("is_equal") {
    REQUIRE_IS_EQUAL(i8(0), u8(0));
    REQUIRE_IS_EQUAL(i16(123), u16(123));
    REQUIRE_IS_EQUAL(i32(-100), i64(-100));
    REQUIRE_IS_EQUAL(u32(3000), i64(3000));
    REQUIRE_IS_EQUAL(f32(2.5f), f64(2.5));
    REQUIRE_IS_EQUAL(f64(-1.0), f64(-1.0L));
    REQUIRE_IS_EQUAL(u64(0ULL), i64(0LL));
    REQUIRE_IS_EQUAL(i32(std::numeric_limits<i32>::max()), i64(std::numeric_limits<i32>::max()));
    REQUIRE_IS_EQUAL(
        f32(std::numeric_limits<f32>::infinity()),
        f64(std::numeric_limits<f64>::infinity())
    );
    REQUIRE_IS_EQUAL(f64(-0.0), f32(-0.0f));
    REQUIRE_IS_EQUAL(i8(-128), i16(-128));
    REQUIRE_IS_EQUAL(u16(65535), u32(65535));
    REQUIRE_IS_EQUAL(i64(-1), f64(-1.0L));
    REQUIRE_IS_EQUAL(f64(0.0L), f64(0.0));
    REQUIRE_IS_EQUAL(u8(255), i32(255));
    REQUIRE_IS_EQUAL(i32(2147483647), f64(2147483647.0));
    REQUIRE_IS_EQUAL(u32(1), f64(1.0));
    REQUIRE_IS_EQUAL(f32(0.5f), f64(0.5L));
    REQUIRE_IS_EQUAL((unsigned long long)(42), f64(42.0L));
    REQUIRE_IS_EQUAL(i16(-32768), i32(-32768));
    REQUIRE_IS_EQUAL(i64(1234567890123LL), u64(1234567890123ULL));
    REQUIRE_IS_EQUAL(u32(0), f32(0.0f));
    REQUIRE_IS_EQUAL(
        f64(-std::numeric_limits<f64>::infinity()),
        f64(-std::numeric_limits<f64>::infinity())
    );
    REQUIRE_IS_EQUAL(f64(123456.0L), f64(123456.0L));

    REQUIRE_NOT_EQUAL(i8(0), u8(1));
    REQUIRE_NOT_EQUAL(i16(123), u16(124));
    REQUIRE_NOT_EQUAL(i32(-100), i64(100));
    REQUIRE_NOT_EQUAL(u32(3000), i64(-3000));
    REQUIRE_NOT_EQUAL(f32(2.5f), f64(2.5001));
    REQUIRE_NOT_EQUAL(f64(-1.0), f64(-1.0000000001L));
    REQUIRE_NOT_EQUAL(u64(1), i64(2));
    REQUIRE_NOT_EQUAL(
        i32(std::numeric_limits<i32>::max()),
        i64(std::numeric_limits<i32>::max()) - 1
    );
    REQUIRE_NOT_EQUAL(
        u64(std::numeric_limits<u64>::max()),
        f64(std::numeric_limits<u64>::max()) - 1.0
    );
    REQUIRE_NOT_EQUAL(u64(std::numeric_limits<u64>::max()), f64(std::numeric_limits<u64>::max()));
    REQUIRE_NOT_EQUAL(
        f32(std::numeric_limits<f32>::infinity()),
        f64(std::numeric_limits<f64>::lowest())
    );
    REQUIRE_NOT_EQUAL(f64(123.0), f64(124.0));
    REQUIRE_NOT_EQUAL(f64(1.0L), f64(1.0000000001L));
    REQUIRE_NOT_EQUAL(i8(-128), i16(-127));
    REQUIRE_NOT_EQUAL(u16(65534), u32(65535));
    REQUIRE_NOT_EQUAL(i64(-1), f64(1.0L));
    REQUIRE_NOT_EQUAL(f64(0.0L), f64(0.1));
    REQUIRE_NOT_EQUAL(u8(255), i32(254));
    REQUIRE_NOT_EQUAL(i32(2147483647), f64(2147483646.0));
    REQUIRE_NOT_EQUAL(u32(1), f64(2.0));
    REQUIRE_NOT_EQUAL(f32(0.5f), f64(0.5000001L));
    REQUIRE_NOT_EQUAL((unsigned long long)(42), f64(42.1));
    REQUIRE_NOT_EQUAL(i16(-32768), i32(-32767));
    REQUIRE_NOT_EQUAL(i64(1234567890123LL), u64(1234567890124ULL));
    REQUIRE_NOT_EQUAL(u32(0), f32(0.0001f));

    REQUIRE_FALSE(is_equal(std::numeric_limits<f64>::quiet_NaN(), f64(0.0)));
    REQUIRE_FALSE(is_equal(std::numeric_limits<f64>::quiet_NaN(), i32(0)));
    REQUIRE_FALSE(
        is_equal(std::numeric_limits<f64>::quiet_NaN(), std::numeric_limits<f32>::quiet_NaN())
    );
}

TEST_CASE("is_convertible") {
    REQUIRE(is_convertible<i32>(f64(0.0)));
    REQUIRE(is_convertible<i32>(f64(2147483647.0)));
    REQUIRE(is_convertible<u32>(i32(0)));
    REQUIRE(is_convertible<u32>(i64(4294967295LL)));
    REQUIRE(is_convertible<i16>(i8(127)));
    REQUIRE(is_convertible<u8>(i32(255)));
    REQUIRE(is_convertible<i64>(f64(9007199254740992.0)));
    REQUIRE(is_convertible<f32>(i32(123456)));
    REQUIRE(is_convertible<f64>(f32(1.5f)));
    REQUIRE(is_convertible<f32>(f64(1.5)));
    REQUIRE(is_convertible<u64>(f64(0.0)));
    REQUIRE(is_convertible<i32>(f32(-100.0f)));
    REQUIRE(is_convertible<i8>(i16(-128)));
    REQUIRE(is_convertible<u16>(u32(65535)));
    REQUIRE(is_convertible<i32>(i32(std::numeric_limits<i32>::min())));
    REQUIRE(is_convertible<i64>(f64(1.0L)));
    REQUIRE(is_convertible<f64>(std::numeric_limits<i32>::min()));
    REQUIRE(is_convertible<i32>(f64(-2147483648.0)));
    REQUIRE(is_convertible<i32>(u32(100)));
    REQUIRE(is_convertible<i64>(f64(9007199254740991.0)));
    REQUIRE(is_convertible<i32>(f32(16777216.0f)));
    REQUIRE(is_convertible<u8>(f64(0.0)));
    REQUIRE(is_convertible<i16>(f64(-32768.0)));
    REQUIRE(is_convertible<i64>(u64(12345ULL)));
    REQUIRE(is_convertible<u64>(i64(0)));
    REQUIRE(is_convertible<f32>(std::numeric_limits<f64>::infinity()));
    REQUIRE(is_convertible<f64>(1.0 + std::numeric_limits<f32>::epsilon()));

    REQUIRE_FALSE(is_convertible<i32>(f64(0.1)));
    REQUIRE_FALSE(is_convertible<i32>(f64(2147483648.0)));
    REQUIRE_FALSE(is_convertible<u32>(i32(-1)));
    REQUIRE_FALSE(is_convertible<i8>(i16(128)));
    REQUIRE_FALSE(is_convertible<u8>(f64(255.5)));
    REQUIRE_FALSE(is_convertible<i16>(f64(40000.0)));
    REQUIRE_FALSE(is_convertible<i32>(f64(std::numeric_limits<f64>::infinity())));
    REQUIRE_FALSE(is_convertible<i32>(f64(std::numeric_limits<f64>::quiet_NaN())));
    REQUIRE_FALSE(is_convertible<i64>(f64(9.223372036854775808e18)));
    REQUIRE_FALSE(is_convertible<u32>(f64(-0.1)));
    REQUIRE_FALSE(is_convertible<u16>(i32(70000)));
    REQUIRE_FALSE(is_convertible<i32>(f32(2147483648.0f)));
    REQUIRE_FALSE(is_convertible<i16>(f64(-32769.0)));
    REQUIRE_FALSE(is_convertible<u8>(i32(256)));
    REQUIRE_FALSE(is_convertible<u8>(i32(-1)));
    REQUIRE_FALSE(is_convertible<u64>(f64(-1.0)));
    REQUIRE_FALSE(is_convertible<i32>(f64(1.5L)));
    REQUIRE_FALSE(is_convertible<i64>(f64(std::numeric_limits<f64>::max())));
    REQUIRE_FALSE(is_convertible<i32>(f64(-2147483649.0)));
    REQUIRE_FALSE(is_convertible<u32>(f64(std::numeric_limits<f64>::infinity())));
    REQUIRE_FALSE(is_convertible<i32>(f32(1.0000001f)));
    REQUIRE_FALSE(is_convertible<i32>(f64(-0.00001)));
    REQUIRE_FALSE(is_convertible<u8>(f64(std::numeric_limits<f64>::quiet_NaN())));
    REQUIRE_FALSE(is_convertible<i32>(f64(1e20)));
    REQUIRE_FALSE(is_convertible<i16>(f32(32768.0f)));
    REQUIRE_FALSE(is_convertible<f32>(1.0 + std::numeric_limits<f64>::epsilon()));
    REQUIRE_FALSE(is_convertible<i32>(1.0f + std::numeric_limits<f32>::epsilon()));
    REQUIRE_FALSE(is_convertible<u32>(std::numeric_limits<f64>::epsilon()));
}

TEST_CASE("in_range") {
    REQUIRE(in_range(u32(5), i32(10)));
    REQUIRE(in_range(u32(8), std::numeric_limits<f64>::infinity()));
    REQUIRE(in_range(u32(5), f64(100.0)));
    REQUIRE(in_range(u32(0), i32(1)));
    REQUIRE(in_range(i64(72), u8(255)));
    REQUIRE(in_range(u32(100), f64(100.000001)));
    REQUIRE(in_range(std::numeric_limits<i32>::max(), std::numeric_limits<u32>::max()));

    REQUIRE_FALSE(in_range(u32(5), i32(-10)));
    REQUIRE_FALSE(in_range(std::numeric_limits<f64>::infinity(), i8(72)));
    REQUIRE_FALSE(in_range(std::numeric_limits<f64>::quiet_NaN(), 100));
    REQUIRE_FALSE(in_range(u32(100), f64(99.99)));
    REQUIRE_FALSE(in_range(u32(100), f64(99.99)));
    REQUIRE_FALSE(in_range(u32(1), i32(0)));
    REQUIRE_FALSE(in_range(u8(255), i64(72)));
    REQUIRE_FALSE(in_range(std::numeric_limits<u32>::max(), std::numeric_limits<i32>::max()));
}

TEST_CASE("checked_cast") {
    REQUIRE_NOTHROW(checked_cast<u64>(std::numeric_limits<i32>::max()));
    REQUIRE_NOTHROW(checked_cast<u64>(std::numeric_limits<u32>::max()));
    REQUIRE_NOTHROW(checked_cast<u64>(std::numeric_limits<i64>::max()));
    REQUIRE_NOTHROW(checked_cast<u64>(std::numeric_limits<u64>::max()));

    REQUIRE_NOTHROW(checked_cast<i64>(std::numeric_limits<i32>::max()));
    REQUIRE_NOTHROW(checked_cast<i64>(std::numeric_limits<u32>::max()));
    REQUIRE_NOTHROW(checked_cast<i64>(std::numeric_limits<i64>::max()));
    REQUIRE_THROWS(checked_cast<i64>(std::numeric_limits<u64>::max()));

    REQUIRE_NOTHROW(checked_cast<u32>(std::numeric_limits<i32>::max()));
    REQUIRE_NOTHROW(checked_cast<u32>(std::numeric_limits<u32>::max()));
    REQUIRE_THROWS(checked_cast<u32>(std::numeric_limits<i64>::max()));
    REQUIRE_THROWS(checked_cast<u32>(std::numeric_limits<u64>::max()));

    REQUIRE_NOTHROW(checked_cast<i32>(std::numeric_limits<i32>::max()));
    REQUIRE_THROWS(checked_cast<i32>(std::numeric_limits<u32>::max()));
    REQUIRE_THROWS(checked_cast<i32>(std::numeric_limits<i64>::max()));
    REQUIRE_THROWS(checked_cast<i32>(std::numeric_limits<u64>::max()));

    REQUIRE_THROWS(checked_cast<i8>(i32(200)));
    REQUIRE_THROWS(checked_cast<i8>(f64(200.0)));
    REQUIRE_THROWS(checked_cast<u8>(i64(-1337)));

    REQUIRE_NOTHROW(checked_cast<f64>(i32(200)));
    REQUIRE_NOTHROW(checked_cast<u8>(f64(201.0)));
    REQUIRE_NOTHROW(checked_cast<u8>(i32(202)));
}