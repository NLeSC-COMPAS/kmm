#include <sstream>
#include <string>

#include "catch2/catch_all.hpp"

#include "kmm/utils/fixed_vector.hpp"

using namespace kmm;

TEST_CASE("fixed_vector") {
    SECTION("size/alignment") {
        REQUIRE(sizeof(fixed_vector<int32_t, 0>) == 1);
        REQUIRE(sizeof(fixed_vector<int32_t, 1>) == 4);
        REQUIRE(sizeof(fixed_vector<int32_t, 2>) == 8);
        REQUIRE(sizeof(fixed_vector<int32_t, 3>) == 16);
        REQUIRE(sizeof(fixed_vector<int32_t, 4>) == 16);
        REQUIRE(sizeof(fixed_vector<int32_t, 5>) == 32);
        REQUIRE(sizeof(fixed_vector<int32_t, 6>) == 32);
        REQUIRE(sizeof(fixed_vector<int32_t, 7>) == 32);
        REQUIRE(sizeof(fixed_vector<int32_t, 8>) == 32);

        REQUIRE(alignof(fixed_vector<int32_t, 0>) == 1);
        REQUIRE(alignof(fixed_vector<int32_t, 1>) == 4);
        REQUIRE(alignof(fixed_vector<int32_t, 2>) == 8);
        REQUIRE(alignof(fixed_vector<int32_t, 3>) == 16);
        REQUIRE(alignof(fixed_vector<int32_t, 4>) == 16);
        REQUIRE(alignof(fixed_vector<int32_t, 5>) == 16);
        REQUIRE(alignof(fixed_vector<int32_t, 6>) == 16);
        REQUIRE(alignof(fixed_vector<int32_t, 7>) == 16);
        REQUIRE(alignof(fixed_vector<int32_t, 8>) == 16);
    }

    SECTION("N=0") {
        fixed_vector<int, 0> a;
        fixed_vector<std::string, 0> b;

        (void)a;
        (void)b;
    }

    SECTION("N=1") {
        fixed_vector<int, 1> a;
        REQUIRE(a.x == 0);

        fixed_vector<std::string, 1> b = {"a"};
        REQUIRE(b.x == "a");
    }

    SECTION("N=2") {
        fixed_vector<int, 2> a;
        REQUIRE(a.x == 0);
        REQUIRE(a.y == 0);

        fixed_vector<std::string, 2> b = {"a", "b"};
        REQUIRE(b.x == "a");
        REQUIRE(b.y == "b");

        REQUIRE(b[0] == "a");
        REQUIRE(b[1] == "b");
    }

    SECTION("N=3") {
        fixed_vector<int, 3> a;
        REQUIRE(a.x == 0);
        REQUIRE(a.y == 0);
        REQUIRE(a.z == 0);

        fixed_vector<std::string, 3> b = {"a", "b", "c"};
        REQUIRE(b.x == "a");
        REQUIRE(b.y == "b");
        REQUIRE(b.z == "c");

        REQUIRE(b[0] == "a");
        REQUIRE(b[1] == "b");
        REQUIRE(b[2] == "c");
    }

    SECTION("N=4") {
        fixed_vector<int, 4> a;
        REQUIRE(a.x == 0);
        REQUIRE(a.y == 0);
        REQUIRE(a.z == 0);
        REQUIRE(a.w == 0);

        fixed_vector<std::string, 4> b = {"a", "b", "c", "d"};
        REQUIRE(b.x == "a");
        REQUIRE(b.y == "b");
        REQUIRE(b.z == "c");
        REQUIRE(b.w == "d");

        REQUIRE(b[0] == "a");
        REQUIRE(b[1] == "b");
        REQUIRE(b[2] == "c");
        REQUIRE(b[3] == "d");
    }

    SECTION("N=5") {
        fixed_vector<int, 5> a;
        REQUIRE(a[0] == 0);
        REQUIRE(a[1] == 0);
        REQUIRE(a[2] == 0);
        REQUIRE(a[3] == 0);
        REQUIRE(a[4] == 0);

        fixed_vector<std::string, 5> b = {"a", "b", "c", "d", "e"};
        REQUIRE(b[0] == "a");
        REQUIRE(b[1] == "b");
        REQUIRE(b[2] == "c");
        REQUIRE(b[3] == "d");
        REQUIRE(b[4] == "e");
    }

    SECTION("operator==") {
        fixed_vector<int, 2> a = {-1, 1};
        fixed_vector<int, 3> b = {-1, 1, 2};
        fixed_vector<float, 3> c = {-1, 1, 2};
        fixed_vector<unsigned int, 3> d = {uint(-1), 1, 2};

        REQUIRE(a == a);
        REQUIRE(a != b);
        REQUIRE(a != c);
        REQUIRE(a != d);

        REQUIRE(b != a);
        REQUIRE(b == b);
        REQUIRE(b == c);
        REQUIRE(b != d);

        REQUIRE(c != a);
        REQUIRE(c == b);
        REQUIRE(c == c);
        REQUIRE(c != d);

        REQUIRE(d != a);
        REQUIRE(d != b);
        REQUIRE(d != c);
        REQUIRE(d == d);
    }

    SECTION("operator<<") {
        fixed_vector<int, 5> a = {1, 2, 3, 4, 5};

        std::stringstream stream;
        stream << a;
        REQUIRE(stream.str() == "{1, 2, 3, 4, 5}");
    }

    SECTION("concat") {
        fixed_vector<int, 3> a = {1, 2, 3};
        fixed_vector<int, 2> b = {4, 5};
        fixed_vector<int, 5> c = concat(a, b);

        REQUIRE(c[0] == 1);
        REQUIRE(c[1] == 2);
        REQUIRE(c[2] == 3);
        REQUIRE(c[3] == 4);
        REQUIRE(c[4] == 5);
    }
}