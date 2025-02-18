#include "catch2/catch_all.hpp"

#include "kmm/core/range.hpp"

using namespace kmm;

TEST_CASE("Range basics") {
    Range<> a {0, 0};
    CHECK(a.begin == 0);
    CHECK(a.end == 0);

    Range<> b {3};
    CHECK(b.begin == 0);
    CHECK(b.end == 3);

    Range<> c {1, 5};
    CHECK(c.begin == 1);
    CHECK(c.end == 5);

    Range<> d {Range<char> {3, 8}};
    CHECK(d.begin == 3);
    CHECK(d.end == 8);

    SECTION("contains index") {
        CHECK_FALSE(a.contains(-1));
        CHECK_FALSE(a.contains(0));
        CHECK_FALSE(a.contains(1));
        CHECK_FALSE(a.contains(5));

        CHECK_FALSE(b.contains(-1));
        CHECK(b.contains(0));
        CHECK(b.contains(1));
        CHECK_FALSE(b.contains(5));

        CHECK_FALSE(c.contains(-1));
        CHECK_FALSE(c.contains(0));
        CHECK(c.contains(1));
        CHECK_FALSE(c.contains(5));

        CHECK_FALSE(d.contains(-1));
        CHECK_FALSE(d.contains(0));
        CHECK_FALSE(d.contains(1));
        CHECK(d.contains(5));
    }

    SECTION("contains range") {
        CHECK(a.contains(a));
        CHECK_FALSE(a.contains(b));
        CHECK_FALSE(a.contains(c));
        CHECK_FALSE(a.contains(d));

        CHECK(b.contains(a));
        CHECK(b.contains(b));
        CHECK_FALSE(b.contains(c));
        CHECK_FALSE(b.contains(d));

        CHECK_FALSE(c.contains(a));
        CHECK_FALSE(c.contains(b));
        CHECK(c.contains(c));
        CHECK_FALSE(c.contains(d));

        CHECK_FALSE(d.contains(a));
        CHECK_FALSE(d.contains(b));
        CHECK_FALSE(d.contains(c));
        CHECK(d.contains(d));
    }

    SECTION("overlaps range") {
        CHECK_FALSE(a.overlaps(a));
        CHECK_FALSE(a.overlaps(b));
        CHECK_FALSE(a.overlaps(c));
        CHECK_FALSE(a.overlaps(d));

        CHECK_FALSE(b.overlaps(a));
        CHECK(b.overlaps(b));
        CHECK(b.overlaps(c));
        CHECK_FALSE(b.overlaps(d));

        CHECK_FALSE(c.overlaps(a));
        CHECK(c.overlaps(b));
        CHECK(c.overlaps(c));
        CHECK(c.overlaps(d));

        CHECK_FALSE(d.overlaps(a));
        CHECK_FALSE(d.overlaps(b));
        CHECK(d.overlaps(c));
        CHECK(d.overlaps(d));
    }

    SECTION("size") {
        CHECK(a.size() == 0);
        CHECK(b.size() == 3);
        CHECK(c.size() == 4);
        CHECK(d.size() == 5);
    }

    SECTION("is_empty") {
        CHECK(a.is_empty());
        CHECK_FALSE(b.is_empty());
        CHECK_FALSE(c.is_empty());
        CHECK_FALSE(d.is_empty());
    }

    SECTION("split_off") {
        auto x = d.split_off(5);
        CHECK(x.begin == 5);
        CHECK(x.end == 8);

        CHECK(d.begin == 3);
        CHECK(d.end == 5);
    }
}

TEST_CASE("WorkBounds constructor") {
    SECTION("default") {
        auto a = WorkBounds {};
        CHECK(a.x.begin == 0);
        CHECK(a.y.begin == 0);
        CHECK(a.z.begin == 0);
        CHECK(a.x.end == 1);
        CHECK(a.y.end == 1);
        CHECK(a.z.end == 1);
    }

    SECTION("number") {
        auto a = WorkBounds {3, 5};
        CHECK(a.x.begin == 0);
        CHECK(a.y.begin == 0);
        CHECK(a.z.begin == 0);
        CHECK(a.x.end == 3);
        CHECK(a.y.end == 5);
        CHECK(a.z.end == 1);
    }

    SECTION("range") {
        auto a = WorkBounds {Range {1, 5}, 3};
        CHECK(a.x.begin == 1);
        CHECK(a.y.begin == 0);
        CHECK(a.z.begin == 0);
        CHECK(a.x.end == 5);
        CHECK(a.y.end == 3);
        CHECK(a.z.end == 1);
    }

    SECTION("offset and size") {
        auto a = WorkBounds::from_offset_size(Index {1, 2}, Dim {3, 8});
        CHECK(a.x.begin == 1);
        CHECK(a.y.begin == 2);
        CHECK(a.z.begin == 0);
        CHECK(a.x.end == 4);
        CHECK(a.y.end == 10);
        CHECK(a.z.end == 1);
    }
}