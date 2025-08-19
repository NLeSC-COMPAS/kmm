#include "catch2/catch_all.hpp"

#include "kmm/utils/dim.hpp"

using namespace kmm;

TEST_CASE("Dim") {
    SECTION("constructor") {
        Dim<3> a;
        Dim<3> b = {-1, 1, 2};

        REQUIRE(a[0] == 1);
        REQUIRE(a[1] == 1);
        REQUIRE(a[2] == 1);

        REQUIRE(b[0] == -1);
        REQUIRE(b[1] == 1);
        REQUIRE(b[2] == 2);
    }

    SECTION("conversions") {
        Dim<3> a = {1, 2, 3};
        Dim<3> b = {-1, 1, 1};

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

        REQUIRE_THROWS(Dim<2> {a});
        REQUIRE(Dim<4> {a} == Dim<4> {1, 2, 3, 1});
        REQUIRE_THROWS(Dim<2, unsigned int> {a});
        REQUIRE(Dim<4, unsigned int> {a} == Dim<4, unsigned int> {1, 2, 3, 1});

        REQUIRE(Dim<2> {b} == Dim<2> {-1, 1});
        REQUIRE(Dim<4> {b} == Dim<4> {-1, 1, 1, 1});
        REQUIRE_THROWS(Dim<2, unsigned int> {b});
        REQUIRE_THROWS(Dim<4, unsigned int> {b});

        REQUIRE(Dim<2>::from(a) == Dim<2> {1, 2});
        REQUIRE(Dim<4>::from(a) == Dim<4> {1, 2, 3, 1});
        REQUIRE(Dim<2, unsigned int>::from(a) == Dim<2, unsigned int> {1, 2});
        REQUIRE(Dim<4, unsigned int>::from(a) == Dim<4, unsigned int> {1, 2, 3, 1});

        REQUIRE(Dim<2>::from(b) == Dim<2> {-1, 1});
        REQUIRE(Dim<4>::from(b) == Dim<4> {-1, 1, 1, 1});
        REQUIRE(Dim<2, unsigned int>::from(b) == Dim<2, unsigned int> {UINT_MAX, 1});
        REQUIRE(Dim<4, unsigned int>::from(b) == Dim<4, unsigned int> {UINT_MAX, 1, 1, 1});
    }

    SECTION("fill") {
        auto fill = Dim<3>::fill(1337);
        auto one = Dim<3>::one();
        auto zero = Dim<3>::zero();

        REQUIRE(fill == Dim<3> {1337, 1337, 1337});
        REQUIRE(one == Dim<3> {1, 1, 1});
        REQUIRE(zero == Dim<3> {0, 0, 0});
    }

    SECTION("get_or_default") {
        Dim<3> a;
        Dim<3> b = {-1, 1, 2};

        REQUIRE(a.get_or_default(0) == 1);
        REQUIRE(a.get_or_default(1) == 1);
        REQUIRE(a.get_or_default(2) == 1);
        REQUIRE(a.get_or_default(3) == 1);
        REQUIRE(a.get_or_default(3, 1337) == 1337);

        REQUIRE(b.get_or_default(0) == -1);
        REQUIRE(b.get_or_default(1) == 1);
        REQUIRE(b.get_or_default(2) == 2);
        REQUIRE(b.get_or_default(3) == 1);
        REQUIRE(b.get_or_default(3, 1337) == 1337);
    }

    SECTION("is_empty") {
        Dim<3> a = {1, 2, 3};
        Dim<3> b = {1, -1, 1};
        Dim<3> c = {1, 0, 1};

        REQUIRE(a.is_empty() == false);
        REQUIRE(b.is_empty());
        REQUIRE(c.is_empty());
    }

    SECTION("is_empty") {
        Dim<3> a = {1, 2, 3};
        Dim<3> b = {1, -1, 1};
        Dim<3> c = {1, 0, 1};

        REQUIRE(a.volume() == 6);
        REQUIRE(b.volume() == 0);
        REQUIRE(c.volume() == 0);
    }

    SECTION("contains") {
        Dim<3> a = {0, 0, 0};
        Dim<3> b = {1, 1, 1};
        Dim<3> c = {INT64_MAX, INT64_MAX, INT64_MAX};
        Dim<3> d = {1, -1, 1};

        Point<3> p0 = {0, 0, 0};
        Point<3> p1 = {1, 0, 0};
        Point<3> p2 = {INT64_MAX, INT64_MAX, INT64_MAX};
        Point<3> p3 = {INT64_MAX - 1, INT64_MAX - 1, INT64_MAX - 1};
        Point<2> p4 = {1, 0};
        Point<5> p5 = {1, 2, 3, 0, 0};
        Point<3, unsigned long> p6 = {1, 2, 3};

        REQUIRE_FALSE(a.contains(p0));
        REQUIRE_FALSE(a.contains(p1));
        REQUIRE_FALSE(a.contains(p2));
        REQUIRE_FALSE(a.contains(p3));
        REQUIRE_FALSE(a.contains(p4));
        REQUIRE_FALSE(a.contains(p5));
        REQUIRE_FALSE(a.contains(p6));

        REQUIRE(b.contains(p0));
        REQUIRE_FALSE(b.contains(p1));
        REQUIRE_FALSE(b.contains(p2));
        REQUIRE_FALSE(b.contains(p3));
        REQUIRE_FALSE(b.contains(p4));
        REQUIRE_FALSE(b.contains(p5));
        REQUIRE_FALSE(b.contains(p6));

        REQUIRE(c.contains(p0));
        REQUIRE(c.contains(p1));
        REQUIRE_FALSE(c.contains(p2));
        REQUIRE(c.contains(p3));
        REQUIRE(c.contains(p4));
        REQUIRE(c.contains(p5));
        REQUIRE(c.contains(p6));

        REQUIRE_FALSE(d.contains(p0));
        REQUIRE_FALSE(d.contains(p1));
        REQUIRE_FALSE(d.contains(p2));
        REQUIRE_FALSE(d.contains(p3));
        REQUIRE_FALSE(d.contains(p4));
        REQUIRE_FALSE(d.contains(p5));
        REQUIRE_FALSE(d.contains(p6));
    }

    SECTION("concat") {
        Dim<2> a = {1, 2};
        Dim<3> b = {3, 4, 5};
        Dim<5> c = concat(a, b);

        REQUIRE(c == Dim<5> {1, 2, 3, 4, 5});
    }

    SECTION("operator==") {
        Dim<3> a = {-1, 2, 3};
        Dim<2> b = {-1, 2};
        Dim<3> c = {-1, 2, 1};
        Dim<3, uint64_t> d = {uint64_t(-1), 2, 3};
        Dim<3, double> e = {-1, 2, 3};

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

    SECTION("is_less") {
        REQUIRE(is_less(Dim(int(1)), int(2)));
        REQUIRE(is_less(int(1), Dim(int(2))));
        REQUIRE(is_less(Dim(int(1)), Dim(int(2))));

        REQUIRE(is_less(Dim(int(1)), uint(2)));
        REQUIRE(is_less(int(1), Dim(uint(2))));
        REQUIRE(is_less(Dim(int(1)), Dim(uint(2))));
    }

    SECTION("checked_cast") {
        REQUIRE(checked_cast<int>(Dim(int(1))) == 1);
        REQUIRE(checked_cast<Dim<1, int>>(Dim(int(1))) == Dim(1));
        REQUIRE(checked_cast<Dim<1, int>>(int(1)) == Dim(1));

        REQUIRE(checked_cast<int>(Dim(uint(1))) == 1);
        REQUIRE(checked_cast<Dim<1, int>>(Dim(uint(1))) == Dim(1));
        REQUIRE(checked_cast<Dim<1, int>>(uint(1)) == Dim(1));
    }

    SECTION("operator<<") {
        std::stringstream stream;
        stream << Dim {1, 2, 3};
        REQUIRE(stream.str() == "{1, 2, 3}");
    }
}