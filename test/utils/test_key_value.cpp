#include <limits>
#include <sstream>

#include "catch2/catch_all.hpp"

#include "kmm/utils/key_value.hpp"

using namespace kmm;

TEST_CASE("KeyValue") {
    SECTION("int") {
        KeyValue<int> a {1, 123};
        KeyValue<int> b {2, 123};
        KeyValue<int> c {2, 456};

        REQUIRE(a.key == 1);
        REQUIRE(b.key == 2);
        REQUIRE(c.key == 2);
        REQUIRE(a.value == 123);
        REQUIRE(b.value == 123);
        REQUIRE(c.value == 456);

        REQUIRE(a == a);
        REQUIRE(a != b);
        REQUIRE(a != c);
        REQUIRE(b != a);
        REQUIRE(b == b);
        REQUIRE(b != c);
        REQUIRE(c != a);
        REQUIRE(c != b);
        REQUIRE(c == c);

        REQUIRE_FALSE(a < a);
        REQUIRE(a < b);
        REQUIRE(a < c);
        REQUIRE_FALSE(b < a);
        REQUIRE_FALSE(b < b);
        REQUIRE(b < c);
        REQUIRE_FALSE(c < a);
        REQUIRE_FALSE(c < b);
        REQUIRE_FALSE(c < c);

        REQUIRE(a <= a);
        REQUIRE(a <= b);
        REQUIRE(a <= c);
        REQUIRE_FALSE(b <= a);
        REQUIRE(b <= b);
        REQUIRE(b <= c);
        REQUIRE_FALSE(c <= a);
        REQUIRE_FALSE(c <= b);
        REQUIRE(c <= c);
    }

    SECTION("float") {
        KeyValue<float> a {1, 123.0f};
        KeyValue<float> b {2, 123.0f};
        KeyValue<float> c {3, std::numeric_limits<float>::quiet_NaN()};

        REQUIRE(a.key == 1);
        REQUIRE(b.key == 2);
        REQUIRE(c.key == 3);
        REQUIRE(a.value == 123.0f);
        REQUIRE(b.value == 123.0f);
        REQUIRE(std::isnan(c.value));

        REQUIRE(a == a);
        REQUIRE(a != b);
        REQUIRE(a != c);
        REQUIRE(b != a);
        REQUIRE(b == b);
        REQUIRE_FALSE(b == c);
        REQUIRE(c != a);
        REQUIRE(c != b);
        REQUIRE(c != c);  // because of NaN

        REQUIRE_FALSE(a < a);
        REQUIRE(a < b);
        REQUIRE(a < c);
        REQUIRE_FALSE(b < a);
        REQUIRE_FALSE(b < b);
        REQUIRE(b < c);
        REQUIRE_FALSE(c < a);
        REQUIRE_FALSE(c < b);
        REQUIRE_FALSE(c < c);

        REQUIRE(a <= a);
        REQUIRE(a <= b);
        REQUIRE(a <= c);
        REQUIRE_FALSE(b <= a);
        REQUIRE(b <= b);
        REQUIRE(b <= c);
        REQUIRE_FALSE(c <= a);
        REQUIRE_FALSE(c <= b);
        REQUIRE(c <= c);
    }
}
