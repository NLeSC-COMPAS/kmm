#include "catch2/catch_all.hpp"

#include "kmm/utils/point.hpp"

using namespace kmm;

TEST_CASE("Point") {
    SECTION("constructor") {
        Point<3> a;
        Point<3> b = {-1, 1, 2};

        REQUIRE(a[0] == 0);
        REQUIRE(a[1] == 0);
        REQUIRE(a[2] == 0);

        REQUIRE(b[0] == -1);
        REQUIRE(b[1] == 1);
        REQUIRE(b[2] == 2);
    }

    SECTION("conversions") {
        Point<3> a = {1, 2, 3};
        Point<3> b = {-1, 1, 0};

        REQUIRE_FALSE(a.is_convertible_to<2>());
        REQUIRE(a.is_convertible_to<4>());
        REQUIRE_FALSE(a.is_convertible_to<2, unsigned int>());
        REQUIRE(a.is_convertible_to<3, unsigned int>());
        REQUIRE(a.is_convertible_to<4, unsigned int>());

        REQUIRE(b.is_convertible_to<2>());
        REQUIRE(b.is_convertible_to<4>());
        REQUIRE_FALSE(b.is_convertible_to<2, unsigned int>());
        REQUIRE_FALSE(b.is_convertible_to<3, unsigned int>());
        REQUIRE_FALSE(b.is_convertible_to<4, unsigned int>());

        REQUIRE_THROWS(Point<2> {a});
        REQUIRE(Point<4> {a} == Point<4> {1, 2, 3, 0});
        REQUIRE_THROWS(Point<2, unsigned int> {a});
        REQUIRE(Point<4, unsigned int> {a} == Point<4, unsigned int> {1, 2, 3, 0});

        REQUIRE(Point<2> {b} == Point<2> {-1, 1});
        REQUIRE(Point<4> {b} == Point<4> {-1, 1, 0, 0});
        REQUIRE_THROWS(Point<2, unsigned int> {b});
        REQUIRE_THROWS(Point<4, unsigned int> {b});

        REQUIRE(Point<2>::from(a) == Point<2> {1, 2});
        REQUIRE(Point<4>::from(a) == Point<4> {1, 2, 3, 0});
        REQUIRE(Point<2, unsigned int>::from(a) == Point<2, unsigned int> {1, 2});
        REQUIRE(Point<4, unsigned int>::from(a) == Point<4, unsigned int> {1, 2, 3, 0});

        REQUIRE(Point<2>::from(b) == Point<2> {-1, 1});
        REQUIRE(Point<4>::from(b) == Point<4> {-1, 1, 0, 0});
        REQUIRE(Point<2, unsigned int>::from(b) == Point<2, unsigned int> {UINT_MAX, 1});
        REQUIRE(Point<4, unsigned int>::from(b) == Point<4, unsigned int> {UINT_MAX, 1, 0, 0});
    }

    SECTION("fill") {
        auto fill = Point<3>::fill(1337);
        auto one = Point<3>::one();
        auto zero = Point<3>::zero();

        REQUIRE(fill == Point<3> {1337, 1337, 1337});
        REQUIRE(one == Point<3> {1, 1, 1});
        REQUIRE(zero == Point<3> {0, 0, 0});
    }

    SECTION("get_or_default") {
        Point<3> a;
        Point<3> b = {-1, 1, 2};

        REQUIRE(a.get_or_default(0) == 0);
        REQUIRE(a.get_or_default(1) == 0);
        REQUIRE(a.get_or_default(2) == 0);
        REQUIRE(a.get_or_default(3) == 0);
        REQUIRE(a.get_or_default(3, 1337) == 1337);

        REQUIRE(b.get_or_default(0) == -1);
        REQUIRE(b.get_or_default(1) == 1);
        REQUIRE(b.get_or_default(2) == 2);
        REQUIRE(b.get_or_default(3) == 0);
        REQUIRE(b.get_or_default(3, 1337) == 1337);
    }

    SECTION("operator==") {
        Point<3> a = {-1, 2, 3};
        Point<2> b = {-1, 2};
        Point<3> c = {-1, 2, 0};
        Point<3, uint64_t> d = {uint64_t(-1), 2, 3};
        Point<3, double> e = {-1, 2, 3};

        REQUIRE(a == a);
        REQUIRE(a != b);
        REQUIRE(a != c);
        REQUIRE(a != d);
        REQUIRE(a == e);

        REQUIRE(b != a);
        REQUIRE(b == b);
        REQUIRE(b == c);
        REQUIRE(b != d);
        REQUIRE(b != e);

        REQUIRE(c != a);
        REQUIRE(c == b);
        REQUIRE(c == c);
        REQUIRE(c != d);
        REQUIRE(c != e);

        REQUIRE(d != a);
        REQUIRE(d != b);
        REQUIRE(d != c);
        REQUIRE(d == d);
        REQUIRE(d != e);

        REQUIRE(e == a);
        REQUIRE(e != b);
        REQUIRE(e != c);
        REQUIRE(e != d);
        REQUIRE(e == e);
    }

    SECTION("operator<<") {
        std::stringstream stream;
        stream << Point {1, 2, 3};
        REQUIRE(stream.str() == "{1, 2, 3}");
    }
}