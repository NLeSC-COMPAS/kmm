#include <cmath>

#include "catch2/catch_all.hpp"

#include "kmm/core/distribution.hpp"

using namespace kmm;

TEST_CASE("Distribution<1>") {
    std::vector<ArrayChunk<1>> chunks = {
        ArrayChunk<1> {.owner_id = DeviceId(0), .offset = 0, .size = 10},
        ArrayChunk<1> {.owner_id = DeviceId(1), .offset = 10, .size = 10},
        ArrayChunk<1> {.owner_id = DeviceId(2), .offset = 20, .size = 6}
    };

    SECTION("no swap") {}

    SECTION("swap 0 and 1") {
        std::swap(chunks[0], chunks[1]);
    }

    SECTION("swap 0 and 2") {
        std::swap(chunks[0], chunks[2]);
    }

    SECTION("swap 1 and 2") {
        std::swap(chunks[1], chunks[2]);
    }

    auto dist = Distribution<1>::from_chunks(26, chunks);

    CHECK(dist.num_chunks() == 3);
    CHECK(dist.chunk_size() == Dim {10});
    CHECK(dist.array_size() == Dim {26});

    CHECK(dist.chunk(0).offset == 0);
    CHECK(dist.chunk(0).size == 10);
    CHECK(dist.chunk(0).owner_id == DeviceId(0));

    CHECK(dist.chunk(1).offset == 10);
    CHECK(dist.chunk(1).size == 10);
    CHECK(dist.chunk(1).owner_id == DeviceId(1));

    CHECK(dist.chunk(2).offset == 20);
    CHECK(dist.chunk(2).size == 6);
    CHECK(dist.chunk(2).owner_id == DeviceId(2));
}

TEST_CASE("Distribution<2>") {
    std::vector<ArrayChunk<2>> chunks = {
        ArrayChunk<2> {.owner_id = DeviceId(1), .offset = {0, 0}, .size = {15, 10}},
        ArrayChunk<2> {.owner_id = DeviceId(2), .offset = {0, 10}, .size = {15, 10}},
        ArrayChunk<2> {.owner_id = DeviceId(3), .offset = {0, 20}, .size = {15, 7}},
        ArrayChunk<2> {.owner_id = DeviceId(4), .offset = {15, 0}, .size = {14, 10}},
        ArrayChunk<2> {.owner_id = DeviceId(5), .offset = {15, 10}, .size = {14, 10}},
        ArrayChunk<2> {.owner_id = DeviceId(6), .offset = {15, 20}, .size = {14, 7}}
    };

    SECTION("no swap") {}

    SECTION("swap 0 and 1") {
        std::swap(chunks[0], chunks[1]);
    }

    SECTION("swap 0 and 4") {
        std::swap(chunks[0], chunks[4]);
    }

    SECTION("swap 3 and 5") {
        std::swap(chunks[3], chunks[5]);
    }

    SECTION("swap 3 and 4") {
        std::swap(chunks[3], chunks[4]);
    }

    auto dist = Distribution<2>::from_chunks({29, 27}, chunks);

    CHECK(dist.num_chunks() == 6);
    CHECK(dist.chunk_size() == Dim {15, 10});
    CHECK(dist.array_size() == Dim {29, 27});

    CHECK(dist.chunk(0).offset == Point {0, 0});
    CHECK(dist.chunk(0).size == Dim {15, 10});
    CHECK(dist.chunk(0).owner_id == DeviceId(1));

    CHECK(dist.chunk(1).offset == Point {0, 10});
    CHECK(dist.chunk(1).size == Dim {15, 10});
    CHECK(dist.chunk(1).owner_id == DeviceId(2));

    CHECK(dist.chunk(2).offset == Point {0, 20});
    CHECK(dist.chunk(2).size == Dim {15, 7});
    CHECK(dist.chunk(2).owner_id == DeviceId(3));

    CHECK(dist.chunk(3).offset == Point {15, 0});
    CHECK(dist.chunk(3).size == Dim {14, 10});
    CHECK(dist.chunk(3).owner_id == DeviceId(4));

    CHECK(dist.chunk(4).offset == Point {15, 10});
    CHECK(dist.chunk(4).size == Dim {14, 10});
    CHECK(dist.chunk(4).owner_id == DeviceId(5));

    CHECK(dist.chunk(5).offset == Point {15, 20});
    CHECK(dist.chunk(5).size == Dim {14, 7});
    CHECK(dist.chunk(5).owner_id == DeviceId(6));
}