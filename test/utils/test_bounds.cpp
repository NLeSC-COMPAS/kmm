#include "catch2/catch_all.hpp"

#include "kmm/utils/bounds.hpp"

using namespace kmm;

TEST_CASE("Bounds") {
    SECTION("constructor") {
        Bounds<3> a;
        Bounds<3> b = {1, 2, 3};
        Bounds<3> c = {Range {4, 10}, Range {10}, 7};
        Bounds<3> d = {Range {-1, 1}, Range {1, 2}, Range {2, 3}};

        REQUIRE(a[0] == Range {0, 0});
        REQUIRE(a[1] == Range {0, 0});
        REQUIRE(a[2] == Range {0, 0});

        REQUIRE(b[0] == Range {0, 1});
        REQUIRE(b[1] == Range {0, 2});
        REQUIRE(b[2] == Range {0, 3});

        REQUIRE(c[0] == Range {4, 10});
        REQUIRE(c[1] == Range {0, 10});
        REQUIRE(c[2] == Range {0, 7});

        REQUIRE(d[0] == Range {-1, 1});
        REQUIRE(d[1] == Range {1, 2});
        REQUIRE(d[2] == Range {2, 3});
    }

    SECTION("empty") {
        Bounds<3> e = Bounds<3>::empty();
        REQUIRE(e[0] == Range {0, 0});
        REQUIRE(e[1] == Range {0, 0});
        REQUIRE(e[2] == Range {0, 0});
    }

    SECTION("one") {
        Bounds<3> f = Bounds<3>::one();
        REQUIRE(f[0] == Range {0, 1});
        REQUIRE(f[1] == Range {0, 1});
        REQUIRE(f[2] == Range {0, 1});
    }

    SECTION("from_bounds") {
        auto a = Bounds<3>::from_bounds({1, 2, 3}, {4, 5, 6});
        REQUIRE(a[0] == Range {1, 4});
        REQUIRE(a[1] == Range {2, 5});
        REQUIRE(a[2] == Range {3, 6});
    }

    SECTION("from_offset_size") {
        auto a = Bounds<3>::from_offset_size({1, 2, 3}, {4, 5, 6});
        REQUIRE(a[0] == Range {1, 1 + 4});
        REQUIRE(a[1] == Range {2, 2 + 5});
        REQUIRE(a[2] == Range {3, 3 + 6});
    }

    SECTION("get_or_default") {
        Bounds<3> a = {Range {4, 10}, Range {10}, 7};

        REQUIRE(a.get_or_default(0) == Range {4, 10});
        REQUIRE(a.get_or_default(1) == Range {0, 10});
        REQUIRE(a.get_or_default(2) == Range {0, 7});
        REQUIRE(a.get_or_default(3) == Range {0, 1});
    }

    SECTION("begin/end/size") {
        Bounds<3> a = {Range {4, 10}, Range {10}, 7};

        REQUIRE(a.begin(0) == 4);
        REQUIRE(a.begin(1) == 0);
        REQUIRE(a.begin(2) == 0);
        REQUIRE(a.begin(3) == 0);

        REQUIRE(a.end(0) == 10);
        REQUIRE(a.end(1) == 10);
        REQUIRE(a.end(2) == 7);
        REQUIRE(a.end(3) == 1);

        REQUIRE(a.size(0) == 6);
        REQUIRE(a.size(1) == 10);
        REQUIRE(a.size(2) == 7);
        REQUIRE(a.size(3) == 1);

        REQUIRE(a.begin() == Point {4, 0, 0});
        REQUIRE(a.end() == Point {10, 10, 7});
        REQUIRE(a.size() == Point {6, 10, 7});
    }

    SECTION("is_empty/volume") {
        Bounds<3> a = {1, 1, 1};
        Bounds<3> b = {1, 2, 3};
        Bounds<3> c = {Range {4, 10}, Range {20, 10}, 7};
        Bounds<3> d = {Range {-1, 1}, Range {1, 2}, Range {2, 3}};

        REQUIRE_FALSE(a.is_empty());
        REQUIRE_FALSE(b.is_empty());
        REQUIRE(c.is_empty());
        REQUIRE_FALSE(d.is_empty());

        REQUIRE(a.volume() == 1);
        REQUIRE(b.volume() == 6);
        REQUIRE(c.volume() == 0);
        REQUIRE(d.volume() == 2);
    }

    SECTION("intersection/unite/overlaps/contains") {
        Bounds<3> a = {1, 1, 1};
        Bounds<3> b = {1, 2, 3};
        Bounds<3> c = {Range {4, 10}, Range {20, 10}, 7};
        Bounds<3> d = {Range {-1, 1}, Range {1, 2}, Range {2, 3}};

#define REQUIRE_COMMUTATIVE(X, Y)                          \
    REQUIRE((X).intersection((Y)) == (Y).intersection(X)); \
    REQUIRE((X).unite((Y)) == (Y).unite(X));               \
    REQUIRE((X).overlaps((Y)) == (Y).overlaps(X));

        REQUIRE_COMMUTATIVE(a, a)
        REQUIRE_COMMUTATIVE(a, b)
        REQUIRE_COMMUTATIVE(a, c)
        REQUIRE_COMMUTATIVE(a, d)
        REQUIRE_COMMUTATIVE(b, a)
        REQUIRE_COMMUTATIVE(b, b)
        REQUIRE_COMMUTATIVE(b, c)
        REQUIRE_COMMUTATIVE(b, d)
        REQUIRE_COMMUTATIVE(c, a)
        REQUIRE_COMMUTATIVE(c, b)
        REQUIRE_COMMUTATIVE(c, c)
        REQUIRE_COMMUTATIVE(c, d)
        REQUIRE_COMMUTATIVE(d, a)
        REQUIRE_COMMUTATIVE(d, b)
        REQUIRE_COMMUTATIVE(d, c)
        REQUIRE_COMMUTATIVE(d, d)

        REQUIRE(a.intersection(a) == a);
        REQUIRE(a.intersection(b) == a);
        REQUIRE(a.intersection(c) == Bounds {Range {4, 1}, Range {20, 1}, 1});
        REQUIRE(a.intersection(d) == Bounds {1, Range {1, 1}, Range {2, 1}});
        REQUIRE(b.intersection(a) == a);
        REQUIRE(b.intersection(b) == b);
        REQUIRE(b.intersection(c) == Bounds {Range {4, 1}, Range {20, 2}, 3});
        REQUIRE(b.intersection(d) == Bounds {1, Range {1, 2}, Range {2, 3}});
        REQUIRE(c.intersection(a) == Bounds {Range {4, 1}, Range {20, 1}, 1});
        REQUIRE(c.intersection(b) == Bounds {Range {4, 1}, Range {20, 2}, 3});
        REQUIRE(c.intersection(c) == c);
        REQUIRE(c.intersection(d) == Bounds {Range {4, 1}, Range {20, 2}, Range {2, 3}});
        REQUIRE(d.intersection(a) == Bounds {1, Range {1, 1}, Range {2, 1}});
        REQUIRE(d.intersection(b) == Bounds {1, Range {1, 2}, Range {2, 3}});
        REQUIRE(d.intersection(c) == Bounds {Range {4, 1}, Range {20, 2}, Range {2, 3}});
        REQUIRE(d.intersection(d) == d);

        REQUIRE(a.unite(a) == a);
        REQUIRE(a.unite(b) == b);
        REQUIRE(a.unite(c) == Bounds {10, 10, 7});
        REQUIRE(a.unite(d) == Bounds {Range {-1, 1}, 2, 3});
        REQUIRE(b.unite(a) == b);
        REQUIRE(b.unite(b) == b);
        REQUIRE(b.unite(c) == Bounds {10, 10, 7});
        REQUIRE(b.unite(d) == Bounds {Range {-1, 1}, 2, 3});
        REQUIRE(c.unite(a) == Bounds {10, 10, 7});
        REQUIRE(c.unite(b) == Bounds {10, 10, 7});
        REQUIRE(c.unite(c) == c);
        REQUIRE(c.unite(d) == Bounds {Range {-1, 10}, Range {1, 10}, 7});
        REQUIRE(d.unite(a) == Bounds {Range {-1, 1}, 2, 3});
        REQUIRE(d.unite(b) == Bounds {Range {-1, 1}, 2, 3});
        REQUIRE(d.unite(c) == Bounds {Range {-1, 10}, Range {1, 10}, 7});
        REQUIRE(d.unite(d) == d);

        REQUIRE(a.overlaps(a));
        REQUIRE(a.overlaps(b));
        REQUIRE_FALSE(a.overlaps(c));
        REQUIRE_FALSE(a.overlaps(d));
        REQUIRE(b.overlaps(a));
        REQUIRE(b.overlaps(b));
        REQUIRE_FALSE(b.overlaps(c));
        REQUIRE(b.overlaps(d));
        REQUIRE_FALSE(c.overlaps(a));
        REQUIRE_FALSE(c.overlaps(b));
        REQUIRE_FALSE(c.overlaps(c));
        REQUIRE_FALSE(c.overlaps(d));
        REQUIRE_FALSE(d.overlaps(a));
        REQUIRE(d.overlaps(b));
        REQUIRE_FALSE(d.overlaps(c));
        REQUIRE(d.overlaps(d));

        REQUIRE(a.contains(a));
        REQUIRE_FALSE(a.contains(b));
        REQUIRE(a.contains(c));
        REQUIRE_FALSE(a.contains(d));
        REQUIRE(b.contains(a));
        REQUIRE(b.contains(b));
        REQUIRE(b.contains(c));
        REQUIRE_FALSE(b.contains(d));
        REQUIRE_FALSE(c.contains(a));
        REQUIRE_FALSE(c.contains(b));
        REQUIRE(c.contains(c));
        REQUIRE_FALSE(c.contains(d));
        REQUIRE_FALSE(d.contains(a));
        REQUIRE_FALSE(d.contains(b));
        REQUIRE(d.contains(c));
        REQUIRE(d.contains(d));
    }

    SECTION("contains") {
        auto simple = Bounds {10, Range {10, 20}, 30};
        REQUIRE_FALSE(simple.contains(0, 0, 0));
        REQUIRE(simple.contains(0, 10, 10));
        REQUIRE(simple.contains(0, 15, 0));
        REQUIRE(simple.contains(9, 19, 29));
        REQUIRE_FALSE(simple.contains(10, 20, 30));
        REQUIRE_FALSE(simple.contains(-1, 0, 0));

        auto empty = Bounds {10, Range {20, 20}, 30};
        REQUIRE_FALSE(empty.contains(0, 0, 0));
        REQUIRE_FALSE(empty.contains(0, 10, 10));
        REQUIRE_FALSE(empty.contains(10, 20, 30));
        REQUIRE_FALSE(empty.contains(-1, 0, 0));
    }

    SECTION("shift_by") {
        auto x = Bounds {1, 2, 3};

        auto y = x.shift_by(Point {3, 4, 5});
        REQUIRE(y[0] == Range {3, 3 + 1});
        REQUIRE(y[1] == Range {4, 4 + 2});
        REQUIRE(y[2] == Range {5, 5 + 3});

        auto z = y.shift_by(Point {3, 4, 5});
        REQUIRE(z[0] == Range {6, 6 + 1});
        REQUIRE(z[1] == Range {8, 8 + 2});
        REQUIRE(z[2] == Range {10, 10 + 3});
    }

    SECTION("split_tail_along") {
        auto x = Bounds {10, 20, 30};

        SECTION("valid") {
            auto y = x.split_tail_along(1, 10);
            REQUIRE(x == Bounds {10, Range {0, 10}, 30});
            REQUIRE(y == Bounds {10, Range {10, 20}, 30});
        }

        SECTION("invalid axis") {
            auto y = x.split_tail_along(10, 1);
            REQUIRE(x == Bounds {10, 20, 30});
            REQUIRE(y.is_empty());
        }

        SECTION("below start") {
            auto y = x.split_tail_along(1, -10);
            REQUIRE(x == Bounds {10, Range {0, 0}, 30});
            REQUIRE(y == Bounds {10, Range {0, 20}, 30});
        }

        SECTION("after end") {
            auto y = x.split_tail_along(1, 100);
            REQUIRE(x == Bounds {10, Range {0, 20}, 30});
            REQUIRE(y == Bounds {10, Range {20, 20}, 30});
        }
    }

    SECTION("concat") {
        auto x = Bounds {1, 2, 3};
        auto y = Bounds {4, 5};
        REQUIRE(concat(x, y) == Bounds {1, 2, 3, 4, 5});
    }

    SECTION("operator==") {
        auto a = Bounds {3, 2, 1};
        auto b = Bounds {3, 2};
        auto c = Bounds {4, 5, 6};

        REQUIRE(a == a);
        REQUIRE(a == b);
        REQUIRE(a != c);

        REQUIRE(b == a);
        REQUIRE(b == b);
        REQUIRE(b != c);

        REQUIRE(c != a);
        REQUIRE(c != b);
        REQUIRE(c == c);
    }

    SECTION("operator<<") {
        std::stringstream stream;
        stream << Bounds {1, Range {2, 3}, Range {1, -1}};
        REQUIRE(stream.str() == "{0...1, 2...3, 1...-1}");
    }
}