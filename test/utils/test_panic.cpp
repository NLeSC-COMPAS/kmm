#include "catch2/catch_all.hpp"

#include "kmm/utils/panic.hpp"

// Unfortunately, Catch2 does not allow catching SIGABRT. Maybe in the future...
TEST_CASE("panic", "[.shouldskip]") {
    KMM_PANIC("don't worry, this is just a test");
}