#include <iostream>

#include "kmm/api/mapper.hpp"
#include "kmm/api/runtime.hpp"

__global__ void initialize_range(kmm::NDRange chunk, kmm::gpu_subview_mut<float> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;
    if (i >= chunk.x.end) {
        return;
    }

    output[i] = float(i);
}

__global__ void fill_range(kmm::NDRange chunk, float value, kmm::gpu_subview_mut<float> output) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;
    if (i >= chunk.x.end) {
        return;
    }

    output[i] = value;
}

__global__ void vector_add(
    kmm::NDRange range,
    kmm::gpu_subview_mut<float> output,
    kmm::gpu_subview<float> left,
    kmm::gpu_subview<float> right
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.x.begin;

    if (i >= range.x.end) {
        return;
    }

    output[i] = left[i] + right[i];
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int n = 2'000'000'000;
    int chunk_size = n / 10;
    dim3 block_size = 256;

    auto A = kmm::Array<float> {n};
    auto B = kmm::Array<float> {n};
    auto C = kmm::Array<float> {n};

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

    rt.parallel_submit(
        {n},
        {chunk_size},
        kmm::GPUKernel(vector_add, block_size),
        write(C(_x)),
        A(_x),
        B(_x)
    );

    std::vector<float> result(n);
    C.copy_to(result.data(), n);
    // Correctness check
    for (int i = 0; i < n; i++) {
        if (result[i] != float(i) + 1) {
            std::cerr << "Wrong result at " << i << " : " << result[i] << " != " << float(i) + 1 << std::endl;
            return 1;
        }
    }
    std::cout << "Correctness check completed." << std::endl;

    return 0;
}
