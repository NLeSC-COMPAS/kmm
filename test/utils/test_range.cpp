#include <sstream>

#include "catch2/catch_all.hpp"

#include "kmm/utils/range.hpp"

using namespace kmm;

TEST_CASE("range") {
    Range<int> empty;
    REQUIRE(empty.begin == 0);
    REQUIRE(empty.end == 0);

    Range<int> one = {8};
    REQUIRE(one.begin == 0);
    REQUIRE(one.end == 8);

    Range<int> middle = {5, 10};
    REQUIRE(middle.begin == 5);
    REQUIRE(middle.end == 10);

    Range<int> max = {INT_MIN, INT_MAX};
    REQUIRE(max.begin == INT_MIN);
    REQUIRE(max.end == INT_MAX);

    REQUIRE(empty.is_empty());
    REQUIRE_FALSE(one.is_empty());
    REQUIRE_FALSE(middle.is_empty());
    REQUIRE_FALSE(max.is_empty());

    REQUIRE(empty.is_convertible_to<unsigned int>());
    REQUIRE(one.is_convertible_to<unsigned int>());
    REQUIRE(middle.is_convertible_to<unsigned int>());
    REQUIRE_FALSE(max.is_convertible_to<unsigned int>());

    REQUIRE(Range<unsigned int>(empty) == empty);
    REQUIRE(Range<unsigned int>(one) == one);
    REQUIRE(Range<unsigned int>(middle) == middle);
    REQUIRE_THROWS(Range<unsigned int>(max));

    REQUIRE(Range<unsigned int>::from(empty) == empty);
    REQUIRE(Range<unsigned int>::from(one) == one);
    REQUIRE(Range<unsigned int>::from(middle) == middle);
    REQUIRE(Range<unsigned int>::from(max) == Range<unsigned int>(UINT_MAX / 2 + 1, UINT_MAX / 2));

    REQUIRE_FALSE(empty.contains(-1));
    REQUIRE_FALSE(empty.contains(0));
    REQUIRE_FALSE(empty.contains(1));
    REQUIRE_FALSE(empty.contains(-100));
    REQUIRE_FALSE(one.contains(-1));
    REQUIRE(one.contains(0));
    REQUIRE(one.contains(1));
    REQUIRE_FALSE(one.contains(-100));
    REQUIRE_FALSE(middle.contains(-1));
    REQUIRE_FALSE(middle.contains(0));
    REQUIRE_FALSE(middle.contains(1));
    REQUIRE_FALSE(middle.contains(-100));
    REQUIRE(max.contains(-1));
    REQUIRE(max.contains(0));
    REQUIRE(max.contains(1));
    REQUIRE(max.contains(-100));
    REQUIRE(max.contains(INT_MIN));
    REQUIRE_FALSE(max.contains(INT_MAX));

    REQUIRE(empty.contains(empty));
    REQUIRE_FALSE(empty.contains(one));
    REQUIRE_FALSE(empty.contains(middle));
    REQUIRE_FALSE(empty.contains(max));
    REQUIRE(one.contains(empty));
    REQUIRE(one.contains(one));
    REQUIRE_FALSE(one.contains(middle));
    REQUIRE_FALSE(one.contains(max));
    REQUIRE(middle.contains(empty));
    REQUIRE_FALSE(middle.contains(one));
    REQUIRE(middle.contains(middle));
    REQUIRE_FALSE(middle.contains(max));
    REQUIRE(max.contains(empty));
    REQUIRE(max.contains(one));
    REQUIRE(max.contains(middle));
    REQUIRE(max.contains(max));

    REQUIRE_FALSE(empty.overlaps(empty));
    REQUIRE_FALSE(empty.overlaps(one));
    REQUIRE_FALSE(empty.overlaps(middle));
    REQUIRE_FALSE(empty.overlaps(max));
    REQUIRE_FALSE(one.overlaps(empty));
    REQUIRE(one.overlaps(one));
    REQUIRE(one.overlaps(middle));
    REQUIRE(one.overlaps(max));
    REQUIRE_FALSE(middle.overlaps(empty));
    REQUIRE(middle.overlaps(one));
    REQUIRE(middle.overlaps(middle));
    REQUIRE(middle.overlaps(max));
    REQUIRE_FALSE(max.overlaps(empty));
    REQUIRE(max.overlaps(one));
    REQUIRE(max.overlaps(middle));
    REQUIRE(max.overlaps(max));

    REQUIRE(empty.intersection(empty) == empty);
    REQUIRE(empty.intersection(one) == empty);
    REQUIRE(empty.intersection(middle) == Range {5, 0});
    REQUIRE(empty.intersection(max) == empty);
    REQUIRE(one.intersection(empty) == empty);
    REQUIRE(one.intersection(one) == one);
    REQUIRE(one.intersection(middle) == Range {5, 8});
    REQUIRE(one.intersection(max) == one);
    REQUIRE(middle.intersection(empty) == Range {5, 0});
    REQUIRE(middle.intersection(one) == Range {5, 8});
    REQUIRE(middle.intersection(middle) == middle);
    REQUIRE(middle.intersection(max) == middle);
    REQUIRE(max.intersection(empty) == empty);
    REQUIRE(max.intersection(one) == one);
    REQUIRE(max.intersection(middle) == middle);
    REQUIRE(max.intersection(max) == max);

    REQUIRE(empty.size() == 0);
    REQUIRE(one.size() == 8);
    REQUIRE(middle.size() == 5);
    //    REQUIRE(max.size() == -1); // ??

    SECTION("split_tail") {
        auto before = Range {5, 25};
        auto after = before.split_tail(10);
        REQUIRE(before == Range {5, 10});
        REQUIRE(after == Range {10, 25});
    }

    SECTION("shift_by") {
        auto total = Range {5, 25};
        auto shifted = total.shift_by(10);
        REQUIRE(shifted == Range {15, 35});
    }

    SECTION("operator<<") {
        std::stringstream ss;
        ss << middle;
        REQUIRE(ss.str() == "5...10");
    }
}