#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/kmm.hpp"

/// This defines the struct for the host-side code
struct Example {
    int x;
    kmm::Array<float> y;
};

/// This defines the struct for the device-side code
struct ExampleView {
    int x;
    kmm::View<float> y;
};

// This defines the fields of the `Example` struct
KMM_DEFINE_STRUCT_ARGUMENT(Example, it.x, it.y)

// This defines that the "view" of `Example` is `ExampleView`
KMM_DEFINE_STRUCT_VIEW(Example, ExampleView)

void example(kmm::Range<int64_t> range, ExampleView input) {
    KMM_ASSERT(input.x == 123);
    KMM_ASSERT(input.y.size() == 3);
    KMM_ASSERT(input.y[0] == 1.0F);
    KMM_ASSERT(input.y[1] == 2.0F);
    KMM_ASSERT(input.y[2] == 3.0F);
    std::cout << "input is correct for range " << range << "!" << std::endl;
}

int main() {
    using namespace kmm::placeholders;
    auto rt = kmm::make_runtime();
    auto y = rt.allocate({1.0F, 2.0F, 3.0F});
    auto structure = Example {.x = 123, .y = y};

    rt.parallel_submit(kmm::TileDomain(1000, 200), kmm::Host(example), _x, structure);
    rt.synchronize();

    return EXIT_SUCCESS;
}
