#include <set>

#include "catch2/catch_all.hpp"

#include "kmm/core/identifiers.hpp"

using namespace kmm;

TEST_CASE("DeviceStreamSet") {
    auto empty = DeviceStreamSet {};
    auto all = DeviceStreamSet::all();
    auto one = DeviceStreamSet {1};
    auto two = DeviceStreamSet {1, 5};
    auto range = DeviceStreamSet::range(4, 8);

    SECTION("contains") {
        for (size_t i = 0; i < DeviceStreamSet::MAX_SIZE; i++) {
            REQUIRE_FALSE(empty.contains(i));
        }

        for (size_t i = 0; i < DeviceStreamSet::MAX_SIZE; i++) {
            REQUIRE(all.contains(i));
        }

        REQUIRE_FALSE(one.contains(0));
        REQUIRE(one.contains(1));
        REQUIRE_FALSE(one.contains(5));

        REQUIRE_FALSE(two.contains(0));
        REQUIRE(two.contains(1));
        REQUIRE(two.contains(5));

        REQUIRE_FALSE(range.contains(3));
        REQUIRE(range.contains(4));
        REQUIRE(range.contains(5));
        REQUIRE(range.contains(6));
        REQUIRE(range.contains(7));
        REQUIRE_FALSE(range.contains(8));
    }

    SECTION("contains subset") {
        REQUIRE(empty.contains(empty));
        REQUIRE(all.contains(empty));
        REQUIRE(one.contains(empty));
        REQUIRE(two.contains(empty));

        REQUIRE_FALSE(empty.contains(all));
        REQUIRE(all.contains(all));
        REQUIRE_FALSE(one.contains(all));
        REQUIRE_FALSE(two.contains(all));

        REQUIRE_FALSE(empty.contains(one));
        REQUIRE(all.contains(one));
        REQUIRE(one.contains(one));
        REQUIRE(two.contains(one));

        REQUIRE_FALSE(empty.contains(two));
        REQUIRE(all.contains(two));
        REQUIRE_FALSE(one.contains(two));
        REQUIRE(two.contains(two));
    }

    SECTION("operator&") {
        REQUIRE((empty & empty) == empty);
        REQUIRE((all & empty) == empty);
        REQUIRE((one & empty) == empty);
        REQUIRE((two & empty) == empty);
        REQUIRE((range & empty) == empty);

        REQUIRE((empty & all) == empty);
        REQUIRE((all & all) == all);
        REQUIRE((one & all) == one);
        REQUIRE((two & all) == two);
        REQUIRE((range & all) == range);

        REQUIRE((one & one) == one);
        REQUIRE((one & two) == one);
        REQUIRE((one & range) == empty);
        REQUIRE((two & one) == one);
        REQUIRE((two & two) == two);
        REQUIRE((two & range) == DeviceStreamSet {5});
        REQUIRE((range & one) == empty);
        REQUIRE((range & two) == DeviceStreamSet {5});
        REQUIRE((range & range) == range);
    }

    SECTION("operator==") {
        REQUIRE(empty == empty);
        REQUIRE(all == all);
        REQUIRE(one == one);
        REQUIRE(two == two);
        REQUIRE(range == range);

        REQUIRE_FALSE(empty == range);
        REQUIRE_FALSE(all == empty);
        REQUIRE_FALSE(one == all);
        REQUIRE_FALSE(two == one);
        REQUIRE_FALSE(range == two);

        REQUIRE(empty != range);
        REQUIRE(all != empty);
        REQUIRE(one != all);
        REQUIRE(two != one);
        REQUIRE(range != two);
    }
}