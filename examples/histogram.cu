#include <random>

#include "spdlog/spdlog.h"

#include "kmm/kmm.hpp"

void initialize_image(
    unsigned long seed,
    int width,
    int height,
    kmm::SubviewMut<uint8_t, 2> image
) {
    std::mt19937 rand {seed};
    std::uniform_int_distribution<uint8_t> dist {};

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i][j] = dist(rand);
        }
    }
}

void initialize_images(
    kmm::Range<int> subrange,
    int width,
    int height,
    kmm::SubviewMut<uint8_t, 3> images
) {
    for (auto i = subrange.begin; i < subrange.end; i++) {
        initialize_image(i, width, height, images.drop_axis<0>(i));
    }
}

__global__ void calculate_histogram(
    kmm::Range<int> image_ids,
    int width,
    int height,
    kmm::GPUSubview<uint8_t, 3> images,
    kmm::GPUSubviewMut<int, 2> histogram
) {
    int image_id = blockIdx.z * blockDim.z + threadIdx.z + image_ids.begin;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (image_id < int(image_ids.end) && i < height && j < width) {
        uint8_t value = images[image_id][i][j];
        atomicAdd(&histogram[image_id][value], 1);
    }
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int width = 1080;
    int height = 1920;
    int num_images = 2500;
    int images_per_chunk = 500;
    dim3 block_size = 256;

    auto histogram = kmm::Array<int> {{256}};
    auto images = kmm::Array<uint8_t, 3> {{num_images, height, width}};

    auto _imageid = kmm::Axis(2);
    auto _i = kmm::Axis(1);
    auto _j = kmm::Axis(0);

    rt.parallel_submit(
        kmm::TileDomain(num_images, images_per_chunk),
        kmm::Host(initialize_images),
        _imageid,
        width,
        height,
        write(images[_imageid][_][_])
    );

    rt.synchronize();

    rt.parallel_submit(
        kmm::TileDomain({width, height, num_images}, {width, height, images_per_chunk}),
        kmm::GPUKernel(calculate_histogram, block_size),
        _imageid,
        width,
        height,
        images[_imageid][_i][_j],
        reduce(kmm::Reduction::Sum, privatize(_imageid), histogram[_])
    );

    rt.synchronize();

    return 0;
}
