#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/kmm.hpp"

__global__ void initialize_range(kmm::Range<int64_t> range, kmm::GPUSubviewMut<float> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.begin;
    if (i >= range.end) {
        return;
    }

    output[i] = float(i);
}

__global__ void fill_range(
    kmm::Range<int64_t> range,
    float value,
    kmm::GPUSubviewMut<float> output
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.begin;
    if (i >= range.end) {
        return;
    }

    output[i] = value;
}

__global__ void vector_add(
    kmm::Range<int64_t> range,
    kmm::GPUSubviewMut<float> output,
    kmm::GPUSubview<float> left,
    kmm::GPUSubview<float> right
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.begin;

    if (i >= range.end) {
        return;
    }

    output[i] = left[i] + right[i];
}

int main() {
    using namespace kmm::placeholders;

    auto rt = kmm::make_runtime();
    spdlog::set_level(spdlog::level::trace);
    long n = 200'000'000;
    long chunk_size = n / 10;
    dim3 block_size = 256;

    auto A = kmm::Array<float> {n};
    auto B = kmm::Array<float> {n};
    auto C = kmm::Array<float> {n};
    auto domain = kmm::TileDomain(n, chunk_size);

    rt.parallel_submit(  //
        domain,
        kmm::GPUKernel(initialize_range, block_size),
        _x,
        write(A[_x])
    );

    rt.parallel_submit(
        domain,
        kmm::GPUKernel(fill_range, block_size),
        _x,
        float(1.0),
        write(B[_x])
    );

    rt.parallel_submit(
        domain,
        kmm::GPUKernel(vector_add, block_size),
        _x,
        write(C[_x]),
        A[_x],
        B[_x]
    );

    auto result = std::vector<float>(n);
    C.copy_to(result);

    // Correctness check
    for (long i = 0; i < n; i++) {
        if (result[i] != float(i) + 1.0F) {
            std::cerr << "Wrong result at " << i << " : " << result[i] << " != " << float(i) + 1
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Correctness check completed." << std::endl;
    return EXIT_SUCCESS;
}
