#include "kmm/api/access.hpp"
#include "kmm/api/runtime.hpp"

__global__ void initialize_range(
    kmm::rect<1> subrange,
    kmm::cuda_subview_mut<float> output
) {
    int64_t i = blockIdx.x * blockDim.x +  threadIdx.x + subrange.begin();
    if (i >= subrange.end()) {
        return;
    }

    output[i] = float(i);
}

__global__ void fill_range(
    kmm::rect<1> subrange,
    float value,
    kmm::cuda_subview_mut<float> output
) {
    int64_t i = blockIdx.x * blockDim.x +  threadIdx.x + subrange.begin();
    if (i >= subrange.end()) {
        return;
    }

    output[i] = value;
}

__global__ void vector_add(
    kmm::rect<1> subrange,
    kmm::cuda_subview_mut<float> output,
    kmm::cuda_subview<float> left,
    kmm::cuda_subview<float> right
) {
    int64_t i = blockIdx.x * blockDim.x +  threadIdx.x + subrange.begin();
    if (i >= subrange.end()) {
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

    auto A = kmm::Array<float>{n, chunk_size};
    auto B = kmm::Array<float>{n, chunk_size};
    auto C = kmm::Array<float>{n, chunk_size};

    rt.parallel_for(
        {n},
        {chunk_size},
        kmm::CudaKernel(initialize_range, block_size),
        write(A, slice(_x))
    );

    rt.parallel_for(
        {n},
        {chunk_size},
        kmm::CudaKernel(fill_range, block_size),
        float(M_PI),
        write(B, slice(_x))
    );

    rt.parallel_for(
        {n},
        {chunk_size},
        kmm::CudaKernel(vector_add, block_size),
        write(C, slice(_x)),
        read(A, slice(_x)),
        read(B, slice(_x))
    );

    std::vector<float> result(n);
    C.copy_to(result.data());

    return 0;
}
