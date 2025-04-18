#include <iostream>

#include "spdlog/spdlog.h"

#include "kmm/kmm.hpp"

__global__ void initialize_matrix_kernel(
    kmm::Bounds<2, int> chunk,
    kmm::GPUSubviewMut<float, 2> matrix
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + chunk.y.begin;
    int j = blockIdx.x * blockDim.x + threadIdx.x + chunk.x.begin;

    if (i < chunk.y.end && j < chunk.x.end) {
        matrix[i][j] = float(i + 2 * j);
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

bool is_close(float expected, float gotten) {
    return fabsf(expected - gotten) < fmaxf(1e-3F * fabsf(expected), 1e-9F);
}

int run(kmm::RuntimeHandle& rt, int width, int height, int chunk_width, int chunk_height) {
    using namespace kmm::placeholders;
    auto domain = kmm::TileDomain({width, height}, {chunk_width, chunk_height});
    auto matrix = kmm::Array<float, 2> {{height, width}};

    std::cout << "Execute for chunk size: " << chunk_width << "x" << chunk_height << "."
              << std::endl;

    rt.parallel_submit(
        domain,
        kmm::GPUKernel(initialize_matrix_kernel, {16, 16}),
        bounds(_x, _y),
        write(matrix[_y][_x])
    );

    rt.synchronize();

    auto total_sum = kmm::Scalar<float>();
    auto rows_sum = kmm::Array<float>(height);
    auto cols_sum = kmm::Array<float>(width);

    rt.parallel_submit(
        domain,
        kmm::GPUKernel(sum_total_kernel, {16, 16}),
        bounds(_x, _y),
        matrix[_y][_x],
        reduce(kmm::Reduction::Sum, privatize(_y, _x), total_sum)
    );

    rt.synchronize();

    rt.parallel_submit(
        domain,
        kmm::GPUKernel(sum_rows_kernel, {16, 16}),
        bounds(_x, _y),
        matrix[_y][_x],
        reduce(kmm::Reduction::Sum, privatize(_y), rows_sum[_x])
    );

    rt.synchronize();

    rt.parallel_submit(
        domain,
        kmm::GPUKernel(sum_cols_kernel, {16, 16}),
        bounds(_x, _y),
        matrix(_y, _x),
        reduce(kmm::Reduction::Sum, privatize(_x), cols_sum[_y])
    );

    rt.synchronize();

    float total;
    total_sum.copy_to(&total);

    if (!is_close(total, float(1.87125e+08))) {
        std::cerr << "Wrong result for total_sum : " << total << " != " << float(1.87125e+08)
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<float> rows;
    rows_sum.copy_to(rows);

    for (int i = 0; i < height; i++) {
        float expected = (float(width - 1) * 0.5F + float(2 * i)) * float(width);

        if (!is_close(rows[i], expected)) {
            std::cerr << "Wrong result for rows_sum[" << i << "]: " << rows[i] << " != " << expected
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::vector<float> cols;
    cols_sum.copy_to(cols);

    for (int i = 0; i < width; i++) {
        float expected = (float(height - 1) + float(i)) * float(height);

        if (!is_close(cols[i], expected)) {
            std::cerr << "Wrong result for cols_sum[" << i << "]: " << cols[i] << " != " << expected
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

int main() {
    spdlog::set_level(spdlog::level::trace);
    auto rt = kmm::make_runtime();
    int width = 500;
    int height = 500;

    for (int nx = 1; nx <= 8; nx++) {
        for (int ny = 1; ny <= 8; ny++) {
            if (run(rt, width, height, width / nx, height / ny) != EXIT_SUCCESS) {
                return EXIT_FAILURE;
            }
        }
    }

    std::cout << "Correctness check completed." << std::endl;
    return EXIT_SUCCESS;
}