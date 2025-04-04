#include <iostream>
#include <thread>

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

void main_loop(unsigned int id, kmm::RuntimeHandle& rt, long n, long chunk_size, dim3 block_size) {
    using namespace kmm::placeholders;
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
            std::cerr << "[THREAD " << id << "] - wrong result at " << i << " : " << result[i]
                      << " != " << float(i) + 1 << std::endl;
            return;
        }
    }
}

int main() {
    auto rt = kmm::make_runtime();
    spdlog::set_level(spdlog::level::warn);
    long n = 200'000'000;
    long chunk_size = n / 10;
    dim3 block_size = 256;
    unsigned int num_threads = 16;
    std::vector<std::thread> threads;

    for (unsigned int thread = 0; thread < num_threads; thread++) {
        threads.emplace_back(main_loop, thread, std::ref(rt), n, chunk_size, block_size);
    }
    for (unsigned int thread = 0; thread < num_threads; thread++) {
        threads.at(thread).join();
    }

    std::cout << "Correctness check completed." << std::endl;
    return EXIT_SUCCESS;
}
