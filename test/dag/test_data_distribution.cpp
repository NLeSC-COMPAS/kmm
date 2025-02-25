#include <cmath>

#include "catch2/catch_all.hpp"

#include "kmm/dag/data_distribution.hpp"

using namespace kmm;

TEST_CASE("DataDistribution<1>") {
    std::vector<DataChunk<1>> chunks = {
        DataChunk<1> {.owner_id = DeviceId(0), .offset = 0, .size = 10},
        DataChunk<1> {.owner_id = DeviceId(1), .offset = 10, .size = 10},
        DataChunk<1> {.owner_id = DeviceId(2), .offset = 20, .size = 6}};

    std::vector<BufferId> buffers = {BufferId(1), BufferId(2), BufferId(3)};

    SECTION("no swap") {}

    SECTION("swap 0 and 1") {
        std::swap(buffers[0], buffers[1]);
        std::swap(chunks[0], chunks[1]);
    }

    SECTION("swap 0 and 2") {
        std::swap(buffers[0], buffers[2]);
        std::swap(chunks[0], chunks[2]);
    }

    SECTION("swap 1 and 2") {
        std::swap(buffers[1], buffers[2]);
        std::swap(chunks[1], chunks[2]);
    }

    auto dist = DataDistribution<1>::from_chunks(26, chunks, buffers);

    CHECK(buffers[0] == BufferId(1));
    CHECK(buffers[1] == BufferId(2));
    CHECK(buffers[2] == BufferId(3));

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

TEST_CASE("DataDistribution<2>") {
    std::vector<DataChunk<2>> chunks = {
        DataChunk<2> {.owner_id = DeviceId(1), .offset = {0, 0}, .size = {15, 10}},
        DataChunk<2> {.owner_id = DeviceId(2), .offset = {0, 10}, .size = {15, 10}},
        DataChunk<2> {.owner_id = DeviceId(3), .offset = {0, 20}, .size = {15, 7}},
        DataChunk<2> {.owner_id = DeviceId(4), .offset = {15, 0}, .size = {14, 10}},
        DataChunk<2> {.owner_id = DeviceId(5), .offset = {15, 10}, .size = {14, 10}},
        DataChunk<2> {.owner_id = DeviceId(6), .offset = {15, 20}, .size = {14, 7}}};

    std::vector<BufferId> buffers =
        {BufferId(1), BufferId(2), BufferId(3), BufferId(4), BufferId(5), BufferId(6)};

    SECTION("no swap") {}

    SECTION("swap 0 and 1") {
        std::swap(buffers[0], buffers[1]);
        std::swap(chunks[0], chunks[1]);
    }

    SECTION("swap 0 and 4") {
        std::swap(buffers[0], buffers[4]);
        std::swap(chunks[0], chunks[4]);
    }

    SECTION("swap 3 and 5") {
        std::swap(buffers[3], buffers[5]);
        std::swap(chunks[3], chunks[5]);
    }

    SECTION("swap 3 and 4") {
        std::swap(buffers[3], buffers[4]);
        std::swap(chunks[3], chunks[4]);
    }

    auto dist = DataDistribution<2>::from_chunks({29, 27}, chunks, buffers);

    CHECK(buffers[0] == BufferId(1));
    CHECK(buffers[1] == BufferId(2));
    CHECK(buffers[2] == BufferId(3));
    CHECK(buffers[3] == BufferId(4));
    CHECK(buffers[4] == BufferId(5));
    CHECK(buffers[5] == BufferId(6));

    CHECK(dist.num_chunks() == 6);
    CHECK(dist.chunk_size() == Dim {15, 10});
    CHECK(dist.array_size() == Dim {29, 27});

    CHECK(dist.chunk(0).offset == Index {0, 0});
    CHECK(dist.chunk(0).size == Dim {15, 10});
    CHECK(dist.chunk(0).owner_id == DeviceId(1));

    CHECK(dist.chunk(1).offset == Index {0, 10});
    CHECK(dist.chunk(1).size == Dim {15, 10});
    CHECK(dist.chunk(1).owner_id == DeviceId(2));

    CHECK(dist.chunk(2).offset == Index {0, 20});
    CHECK(dist.chunk(2).size == Dim {15, 7});
    CHECK(dist.chunk(2).owner_id == DeviceId(3));

    CHECK(dist.chunk(3).offset == Index {15, 0});
    CHECK(dist.chunk(3).size == Dim {14, 10});
    CHECK(dist.chunk(3).owner_id == DeviceId(4));

    CHECK(dist.chunk(4).offset == Index {15, 10});
    CHECK(dist.chunk(4).size == Dim {14, 10});
    CHECK(dist.chunk(4).owner_id == DeviceId(5));

    CHECK(dist.chunk(5).offset == Index {15, 20});
    CHECK(dist.chunk(5).size == Dim {14, 7});
    CHECK(dist.chunk(5).owner_id == DeviceId(6));
}