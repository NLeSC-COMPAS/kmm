#include "catch2/catch_all.hpp"

#include "kmm/utils/hash_utils.hpp"

using namespace kmm;

TEST_CASE("hash_combine") {
    size_t seed = 0;
    hash_combine(seed, int32_t(32));
    hash_combine(seed, double(32));
    hash_combine(seed, std::string("foo"));
    hash_combine(seed, true);
    REQUIRE(seed != 0);
}

TEST_CASE("hash_combine_range") {
    SECTION("items") {
        size_t seed0 = 0;
        std::array<int, 3> a {1, 2, 3};
        hash_combine_range(seed0, a.data(), a.data() + a.size());

        size_t seed1 = 0;
        hash_combine(seed1, 1);
        hash_combine(seed1, 2);
        hash_combine(seed1, 3);

        REQUIRE(seed0 == seed1);
    }

    SECTION("empty range") {
        size_t seed = 1337;

        std::array<int, 0> a;
        hash_combine_range(seed, a.data(), a.data());

        REQUIRE(seed == 1337);
    }
}