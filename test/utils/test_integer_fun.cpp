#include <cstdint>

#include "catch2/catch_all.hpp"

#include "kmm/utils/integer_fun.hpp"

using namespace kmm;

using i32 = int32_t;
using u64 = uint64_t;

TEST_CASE("div_floor") {
    REQUIRE(div_floor<i32>(0, 5) == 0);
    REQUIRE(div_floor<i32>(8, 4) == 2);
    REQUIRE(div_floor<i32>(-8, 4) == -2);
    REQUIRE(div_floor<i32>(8, -4) == -2);
    REQUIRE(div_floor<i32>(-8, -4) == 2);
    REQUIRE(div_floor<i32>(7, 4) == 1);
    REQUIRE(div_floor<i32>(-7, 4) == -2);
    REQUIRE(div_floor<i32>(7, -4) == -2);
    REQUIRE(div_floor<i32>(-7, -4) == 1);

    REQUIRE(div_floor<u64>(0ULL, 5ULL) == 0ULL);
    REQUIRE(div_floor<u64>(9ULL, 2ULL) == 4ULL);
    REQUIRE(div_floor<u64>(15ULL, 3ULL) == 5ULL);
    REQUIRE(div_floor<u64>(123'456'789ULL, 1'000ULL) == 123'456ULL);
}

TEST_CASE("div_ceil") {
    REQUIRE(div_ceil<i32>(0, 5) == 0);
    REQUIRE(div_ceil<i32>(8, 4) == 2);
    REQUIRE(div_ceil<i32>(-8, 4) == -2);
    REQUIRE(div_ceil<i32>(8, -4) == -2);
    REQUIRE(div_ceil<i32>(-8, -4) == 2);
    REQUIRE(div_ceil<i32>(7, 4) == 2);
    REQUIRE(div_ceil<i32>(-7, 4) == -1);
    REQUIRE(div_ceil<i32>(7, -4) == -1);
    REQUIRE(div_ceil<i32>(-7, -4) == 2);

    REQUIRE(div_ceil<u64>(0ULL, 5ULL) == 0ULL);
    REQUIRE(div_ceil<u64>(9ULL, 2ULL) == 5ULL);
    REQUIRE(div_ceil<u64>(15ULL, 3ULL) == 5ULL);
    REQUIRE(div_ceil<u64>(123'456'789ULL, 1'000ULL) == 123'457ULL);
}

TEST_CASE("round_up_to_multiple") {
    // positive input
    REQUIRE(round_up_to_multiple<i32>(0, 8) == 0);
    REQUIRE(round_up_to_multiple<i32>(7, 4) == 8);
    REQUIRE(round_up_to_multiple<i32>(12, 4) == 12);

    // negative input
    REQUIRE(round_up_to_multiple<i32>(-7, 4) == -4);
    REQUIRE(round_up_to_multiple<i32>(-12, 4) == -12);
    REQUIRE(round_up_to_multiple<i32>(-7, -4) == -4);
    REQUIRE(round_up_to_multiple<i32>(-12, -4) == -12);

    // unsigned
    REQUIRE(round_up_to_multiple<u64>(9ULL, 8ULL) == 16ULL);
    REQUIRE(round_up_to_multiple<u64>(32ULL, 8ULL) == 32ULL);
}

TEST_CASE("round_up_to_power_of_two") {
    REQUIRE(round_up_to_power_of_two<i32>(-1) == 1);
    REQUIRE(round_up_to_power_of_two<i32>(0) == 1);
    REQUIRE(round_up_to_power_of_two<i32>(1) == 1);
    REQUIRE(round_up_to_power_of_two<i32>(2) == 2);
    REQUIRE(round_up_to_power_of_two<i32>(3) == 4);
    REQUIRE(round_up_to_power_of_two<i32>(17) == 32);

    REQUIRE(round_up_to_power_of_two<u64>(63ULL) == 64ULL);
    REQUIRE(round_up_to_power_of_two<u64>(64ULL) == 64ULL);
    REQUIRE(round_up_to_power_of_two<u64>(65ULL) == 128ULL);
}

TEST_CASE("is_power_of_two") {
    REQUIRE(is_power_of_two<i32>(-1) == false);
    REQUIRE(is_power_of_two<i32>(0) == false);
    REQUIRE(is_power_of_two<i32>(1) == true);
    REQUIRE(is_power_of_two<i32>(2) == true);
    REQUIRE(is_power_of_two<i32>(3) == false);
    REQUIRE(is_power_of_two<i32>(16) == true);
    REQUIRE(is_power_of_two<i32>(std::numeric_limits<i32>::min()) == false);
    REQUIRE(is_power_of_two<i32>(std::numeric_limits<i32>::max()) == false);

    REQUIRE(is_power_of_two<u64>(1'024ULL) == true);
    REQUIRE(is_power_of_two<u64>(1'025ULL) == false);
    REQUIRE(is_power_of_two<u64>(std::numeric_limits<u64>::min()) == false);
    REQUIRE(is_power_of_two<u64>(std::numeric_limits<u64>::max()) == false);
}