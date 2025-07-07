#include <sstream>
#include <string>

#include "catch2/catch_all.hpp"

#include "kmm/utils/small_vector.hpp"

using namespace kmm;

struct MyString {
    MyString(std::string x = "<uninitialized>") : value(x) {}
    std::string value;
};

TEST_CASE("small_vector") {
    SECTION("basics") {
        small_vector<int, 4> x;
        REQUIRE(x.capacity() == 4);
        REQUIRE(x.size() == 0);
        REQUIRE(x.is_empty() == true);
        REQUIRE(x.is_heap_allocated() == false);
        REQUIRE(x.begin() == x.data());
        REQUIRE(x.end() == x.data());

        x.push_back(1);

        REQUIRE(x.capacity() == 4);
        REQUIRE(x.size() == 1);
        REQUIRE(x.is_empty() == false);
        REQUIRE(x.is_heap_allocated() == false);
        REQUIRE(x.begin() == x.data());
        REQUIRE(x.end() == x.data() + 1);
        REQUIRE(&x[0] == x.data());
        REQUIRE(x[0] == 1);

        x.push_back(2);
        x.push_back(3);
        x.push_back(4);
        x.push_back(5);

        REQUIRE(x.capacity() == 16);
        REQUIRE(x.size() == 5);
        REQUIRE(x.is_empty() == false);
        REQUIRE(x.is_heap_allocated() == true);
        REQUIRE(x.begin() == x.data());
        REQUIRE(x.end() == x.data() + 5);
        REQUIRE(&x[0] == x.data());
        REQUIRE(x[0] == 1);
        REQUIRE(x[1] == 2);
        REQUIRE(x[2] == 3);
        REQUIRE(x[3] == 4);
        REQUIRE(x[4] == 5);

        x.truncate(2);

        REQUIRE(x.capacity() == 16);
        REQUIRE(x.size() == 2);
        REQUIRE(x.is_empty() == false);
        REQUIRE(x.is_heap_allocated() == true);
        REQUIRE(x.begin() == x.data());
        REQUIRE(x.end() == x.data() + 2);
        REQUIRE(&x[0] == x.data());
        REQUIRE(x[0] == 1);
        REQUIRE(x[1] == 2);

        x.resize(20);

        REQUIRE(x.capacity() == 32);
        REQUIRE(x.size() == 20);
        REQUIRE(x.is_empty() == false);
        REQUIRE(x.is_heap_allocated() == true);
        REQUIRE(x.begin() == x.data());
        REQUIRE(x.end() == x.data() + 20);
        REQUIRE(&x[0] == x.data());
        REQUIRE(x[0] == 1);
        REQUIRE(x[1] == 2);
        REQUIRE(x[2] == 0);
        REQUIRE(x[3] == 0);
        REQUIRE(x[4] == 0);
        REQUIRE(x[5] == 0);
        REQUIRE(x[6] == 0);
        REQUIRE(x[7] == 0);
        REQUIRE(x[19] == 0);

        x.clear();

        REQUIRE(x.capacity() == 32);
        REQUIRE(x.size() == 0);
        REQUIRE(x.is_empty() == true);
        REQUIRE(x.is_heap_allocated() == true);
        REQUIRE(x.begin() == x.data());
        REQUIRE(x.end() == x.data());
    }

    SECTION("constructor") {
        small_vector<std::string, 4> a;  // default
        small_vector<std::string, 4> b = {"a", "b"};  // list
        small_vector<std::string, 4> c = b;  // copy
        small_vector<std::string, 4> d = std::move(c);  //move
        small_vector<std::string, 1> e = d;  // copy, different N
        small_vector<MyString, 4> f = d;  // copy, different T

        REQUIRE(a.size() == 0);

        REQUIRE(b.size() == 2);
        REQUIRE(b[0] == "a");
        REQUIRE(b[1] == "b");

        REQUIRE(c.size() == 0);

        REQUIRE(d.size() == 2);
        REQUIRE(d[0] == "a");
        REQUIRE(d[1] == "b");

        REQUIRE(e.size() == 2);
        REQUIRE(e[0] == "a");
        REQUIRE(e[1] == "b");

        REQUIRE(f.size() == 2);
        REQUIRE(f[0].value == "a");
        REQUIRE(f[1].value == "b");

        // The other two values must be uninitialized
        REQUIRE(f.capacity() == 4);
        REQUIRE(f[2].value == "<uninitialized>");
        REQUIRE(f[3].value == "<uninitialized>");
    }

    SECTION("operator=") {
        small_vector<std::string, 4> a = {"foo", "bar"};
        small_vector<std::string, 4> b;
        small_vector<std::string, 1> c;
        small_vector<MyString, 4> d;
        small_vector<std::string, 4> e;

        b = a;  // operator=(const small_vector&)
        c = a;  // operator=(const small_vector<U, K>&)
        d = a;  // operator=(const small_vector<U, K>&)
        e = std::move(a);  // operator=(small_vector&&)

        REQUIRE(a.size() == 0);

        REQUIRE(b.size() == 2);
        REQUIRE(b[0] == "foo");
        REQUIRE(b[1] == "bar");

        REQUIRE(c.size() == 2);
        REQUIRE(c[0] == "foo");
        REQUIRE(c[1] == "bar");

        REQUIRE(d.size() == 2);
        REQUIRE(d[0].value == "foo");
        REQUIRE(d[1].value == "bar");

        REQUIRE(e.size() == 2);
        REQUIRE(e[0] == "foo");
        REQUIRE(e[1] == "bar");
    }

    SECTION("operator<<") {
        small_vector<int, 4> x = {1, 2, 3, 4, 5, 6};
        small_vector<int, 4> y = {};

        auto stream = std::stringstream();
        stream << x;
        REQUIRE(stream.str() == "{1, 2, 3, 4, 5, 6}");

        stream = std::stringstream();
        stream << y;
        REQUIRE(stream.str() == "{}");
    }
}