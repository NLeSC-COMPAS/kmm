#include "spdlog/spdlog.h"

#include "kmm/kmm.hpp"

__global__ void initialize_matrix_kernel(
    kmm::Bounds<2, int> chunk,
    kmm::GPUSubviewMut<float, 2> matrix
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.y.begin;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;

    if (i < chunk.y.end && j < chunk.x.end) {
        matrix[i][j] = 1.0f;
    }
}

__global__ void sum_total_kernel(
    kmm::Bounds<2, int> chunk,
    kmm::GPUSubview<float, 2> matrix,
    kmm::GPUSubviewMut<float, 2> sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.y.begin;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;

    if (i < chunk.y.end && j < chunk.x.end) {
        sum[i][j] += matrix[i][j];
    }
}

__global__ void sum_rows_kernel(
    kmm::Bounds<2, int> chunk,
    kmm::GPUSubview<float, 2> matrix,
    kmm::GPUSubviewMut<float, 2> rows_sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.y.begin;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;

    if (i < chunk.y.end && j < chunk.x.end) {
        rows_sum[i][j] += matrix[i][j];
    }
}

__global__ void sum_cols_kernel(
    kmm::Bounds<2, int> chunk,
    kmm::GPUSubview<float, 2> matrix,
    kmm::GPUSubviewMut<float, 2> cols_sum
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.y.begin;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;

    if (i < chunk.y.end && j < chunk.x.end) {
        cols_sum[j][i] += matrix[i][j];
    }
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    int width = 32768;
    int height = 32768;
    int chunk_width = width / 8;
    int chunk_height = height / 8;
    auto dist = kmm::TileDomain({width, height}, {chunk_width, chunk_height});

    auto rt = kmm::make_runtime();
    auto matrix = kmm::Array<float, 2> {{height, width}};

    rt.parallel_submit(
        dist,
        kmm::GPUKernel(initialize_matrix_kernel, {16, 16}),
        bounds(_x, _y),
        write(matrix[_y][_x])
    );

    rt.synchronize();

    auto total_sum = kmm::Scalar<float>();
    auto rows_sum = kmm::Array<float>(height);
    auto cols_sum = kmm::Array<float>(width);

    rt.parallel_submit(
        dist,
        kmm::GPUKernel(sum_total_kernel, {16, 16}),
        bounds(_x, _y),
        matrix[_y][_x],
        reduce(kmm::Reduction::Sum, privatize(_y, _x), total_sum)
    );

    rt.synchronize();

    rt.parallel_submit(
        dist,
        kmm::GPUKernel(sum_rows_kernel, {16, 16}),
        bounds(_x, _y),
        matrix[_y][_x],
        reduce(kmm::Reduction::Sum, privatize(_y), rows_sum[_x])
    );

    rt.synchronize();

    rt.parallel_submit(
        dist,
        kmm::GPUKernel(sum_cols_kernel, {16, 16}),
        bounds(_x, _y),
        matrix(_y, _x),
        reduce(kmm::Reduction::Sum, privatize(_x), cols_sum[_y])
    );

    rt.synchronize();

    return EXIT_SUCCESS;
}