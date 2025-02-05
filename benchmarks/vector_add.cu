#include <iostream>
#include <chrono>

#include "kmm/api/mapper.hpp"
#include "kmm/api/runtime.hpp"

using real_type = float;
const unsigned int max_iterations = 10;

__global__ void initialize_range(kmm::NDRange chunk, kmm::gpu_subview_mut<real_type> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;
    if (i >= chunk.x.end) {
        return;
    }

    output[i] = static_cast<real_type>(i);
}

__global__ void fill_range(kmm::NDRange chunk, real_type value, kmm::gpu_subview_mut<real_type> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;
    if (i >= chunk.x.end) {
        return;
    }

    output[i] = value;
}

__global__ void vector_add(
    kmm::NDRange range,
    kmm::gpu_subview_mut<real_type> output,
    kmm::gpu_subview<real_type> left,
    kmm::gpu_subview<real_type> right
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.x.begin;

    if (i >= range.x.end) {
        return;
    }

    output[i] = left[i] + right[i];
}

int main() {
    using namespace kmm::placeholders;

    auto rt = kmm::make_runtime();
    int n = 2'000'000'000;
    unsigned long long int ops = n * max_iterations;
    unsigned long long int mem = (n * 3 * sizeof(real_type)) * max_iterations;
    int chunk_size = n / 10;
    dim3 block_size = 256;
    std::chrono::duration<double> elapsed_time;

    for ( unsigned int iteration = 0; iteration < max_iterations; ++iteration ) {
        auto A = kmm::Array<real_type> {n};
        auto B = kmm::Array<real_type> {n};
        auto C = kmm::Array<real_type> {n};

        // Initialize input arrays
        rt.parallel_submit(
            kmm::Size {n},
            kmm::ChunkPartitioner {chunk_size},
            kmm::GPUKernel(initialize_range, block_size),
            write(A(_x))
        );
        rt.parallel_submit(
            {n},
            {chunk_size},
            kmm::GPUKernel(fill_range, block_size),
            float(1.0),
            write(B(_x))
        );
        rt.synchronize();
        // Benchmark

        auto timing_start = std::chrono::steady_clock::now();
        rt.parallel_submit(
            {n},
            {chunk_size},
            kmm::GPUKernel(vector_add, block_size),
            write(C(_x)),
            A(_x),
            B(_x)
        );
        rt.synchronize();
        auto timing_stop = std::chrono::steady_clock::now();
        elapsed_time += timing_stop - timing_start;

        // Correctness check
        std::vector<real_type> result(n);
        C.copy_to(result.data(), n);
        for (int i = 0; i < n; i++) {
            if (result[i] != static_cast<real_type>(i) + 1) {
                std::cerr << "Wrong result at " << i << " : " << result[i] << " != " << float(i) + 1 << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Total time: " << elapsed_time.count() << " seconds" << std::endl;
    std::cout << "Average iteration time: " << elapsed_time.count() / max_iterations << " seconds" << std::endl;
    std::cout << "Throughput: " << (ops / elapsed_time.count()) / 1'000'000'000 << " GFLOP/s" << std::endl;
    std::cout << "Memory bandwidth: " << (mem / elapsed_time.count()) / 1'000'000'000 << " GB/s" << std::endl;

    return 0;
}
