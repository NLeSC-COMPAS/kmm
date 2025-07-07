#include "catch2/catch_all.hpp"

#include "kmm/utils/checked_math.hpp"

using namespace kmm;

using i8 = int8_t;
using i32 = int32_t;
using u32 = uint32_t;

TEST_CASE("checked_add") {
    REQUIRE(checked_add(i32(5), i32(1)) == i32(6));
    REQUIRE(checked_add(i32(-5), i32(1)) == i32(-4));
    REQUIRE_THROWS(checked_add(std::numeric_limits<i32>::max(), i32(1)));
    REQUIRE_THROWS(checked_add(std::numeric_limits<i32>::min(), i32(-1)));

    REQUIRE(checked_add(u32(5), u32(1)) == u32(6));
    REQUIRE(checked_add(u32(0), u32(1)) == u32(1));
    REQUIRE_THROWS(checked_add(std::numeric_limits<u32>::max(), u32(1)));

    REQUIRE(checked_add(i8(5), i8(1)) == i8(6));
    REQUIRE(checked_add(i8(-5), i8(1)) == i8(-4));
    REQUIRE_THROWS(checked_add(std::numeric_limits<i8>::max(), i8(1)));
    REQUIRE_THROWS(checked_add(std::numeric_limits<i8>::min(), i8(-1)));
}

TEST_CASE("checked_sub") {
    REQUIRE(checked_sub(i32(5), i32(1)) == i32(4));
    REQUIRE(checked_sub(i32(-5), i32(1)) == i32(-6));
    REQUIRE_THROWS(checked_sub(std::numeric_limits<i32>::max(), i32(-1)));
    REQUIRE_THROWS(checked_sub(std::numeric_limits<i32>::min(), i32(1)));

    REQUIRE(checked_sub(u32(5), u32(1)) == u32(4));
    REQUIRE_THROWS(checked_sub(u32(0), u32(1)));
    REQUIRE(checked_sub(std::numeric_limits<u32>::max(), std::numeric_limits<u32>::max()) == 0);

    REQUIRE(checked_sub(i8(5), i8(1)) == i8(4));
    REQUIRE(checked_sub(i8(-5), i8(1)) == i8(-6));
    REQUIRE_THROWS(checked_sub(std::numeric_limits<i8>::max(), i8(-1)));
    REQUIRE_THROWS(checked_sub(std::numeric_limits<i8>::min(), i8(1)));
}

TEST_CASE("checked_mul") {
    REQUIRE(checked_mul(i32(5), i32(1)) == i32(5));
    REQUIRE(checked_mul(i32(-5), i32(1)) == i32(-5));
    REQUIRE(checked_mul(std::numeric_limits<i32>::max(), i32(-1)) == -2147483647);
    REQUIRE_THROWS(checked_mul(std::numeric_limits<i32>::min(), i32(-1)));

    REQUIRE(checked_mul(u32(5), u32(1)) == u32(5));
    REQUIRE(checked_mul(u32(0), u32(1)) == u32(0));
    REQUIRE_THROWS(checked_mul(std::numeric_limits<u32>::max(), std::numeric_limits<u32>::max()));

    REQUIRE(checked_mul(i8(5), i8(-1)) == i8(-5));
    REQUIRE(checked_mul(i8(-5), i8(-1)) == i8(5));
    REQUIRE(checked_mul(std::numeric_limits<i8>::max(), i8(-1)) == -127);
    REQUIRE_THROWS(checked_mul(std::numeric_limits<i8>::min(), i8(-1)));
}

TEST_CASE("checked_sum") {
    SECTION("empty list") {
        std::array<int, 0> values = {};
        REQUIRE(checked_sum(values.data(), values.data()) == 0);
    }

    SECTION("no overflow") {
        std::array<i32, 4> values = {1, 2, 3, 4};
        REQUIRE(checked_sum(values.data(), values.data() + 4) == 10);
    }

    SECTION("overflows") {
        std::array<i32, 4> values = {1, 2, std::numeric_limits<int>::max(), 4};
        REQUIRE_THROWS(checked_sum(values.data(), values.data() + 4));
    }

    SECTION("no overflow with upcast") {
        std::array<i32, 4> values = {1, 2, std::numeric_limits<int>::max(), 4};
        REQUIRE(checked_sum(values.data(), values.data() + 4, uint64_t(1)) == 2147483655LL);
    }
}

TEST_CASE("checked_product") {
    SECTION("empty list") {
        std::array<int, 0> values = {};
        REQUIRE(checked_product(values.data(), values.data()) == 1);
    }

    SECTION("no overflow") {
        std::array<i32, 4> values = {1, 2, 3, 4};
        REQUIRE(checked_product(values.data(), values.data() + 4) == 24);
    }

    SECTION("overflows") {
        std::array<i32, 3> values = {1, std::numeric_limits<int>::max() / 2, 3};
        REQUIRE_THROWS(checked_product(&*values.begin(), &*values.end()));
    }

    SECTION("no overflow with upcast") {
        std::array<i32, 3> values = {1, std::numeric_limits<int>::max() / 2, 3};
        REQUIRE(checked_product(&*values.begin(), &*values.end(), uint64_t(1)) == 3221225469LL);
    }
}