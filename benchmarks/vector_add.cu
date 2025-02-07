#include <iostream>
#include <string>
#include <chrono>

#include "kmm/kmm.hpp"

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

bool inner_loop(kmm::Runtime &rt, unsigned int threads, unsigned long int n, unsigned long int chunk_size, std::chrono::duration<double> &init_time, std::chrono::duration<double> &run_time) {
    using namespace kmm::placeholders;
    dim3 block_size = threads;
    auto timing_start_init = std::chrono::steady_clock::now();
    auto A = kmm::Array<real_type> {n};
    auto B = kmm::Array<real_type> {n};
    auto C = kmm::Array<real_type> {n};

    // Initialize input arrays
    rt.parallel_submit(
        kmm::Dim {n},
        kmm::ChunkPartitioner {chunk_size},
        kmm::GPUKernel(initialize_range, block_size),
        write(A(_x))
    );
    rt.parallel_submit(
        {n},
        {chunk_size},
        kmm::GPUKernel(fill_range, block_size),
        static_cast<real_type>(1.0),
        write(B(_x))
    );
    rt.synchronize();
    auto timing_stop_init = std::chrono::steady_clock::now();
    init_time += timing_stop_init - timing_start_init;

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
    run_time += timing_stop - timing_start;

    // Correctness check
    std::vector<real_type> result(n);
    C.copy_to(result.data(), n);
    for ( unsigned int i = 0; i < n; i++ ) {
        if (result[i] != static_cast<real_type>(i) + 1) {
            std::cerr << "Wrong result at " << i << " : " << result[i] << " != " << static_cast<real_type>(i) + 1.0 << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    auto rt = kmm::make_runtime();
    bool status = false;
    unsigned long int n = 0;
    unsigned int num_threads = 0;
    unsigned int num_chunks = 0;
    double ops = max_iterations;
    double mem = 3.0 * sizeof(real_type) * max_iterations;
    std::chrono::duration<double> init_time, vector_add_time;

    if ( argc != 4 ) {
        std::cerr << "Usage: " << argv[0] << " <threads> <num_chunks> <size>" << std::endl;
        return 1;
    }
    else {
        num_threads = std::stoi(argv[1]);
        num_chunks = std::stoi(argv[2]);
        n = std::stol(argv[3]);
    }
    ops *= n;
    mem *= n;

    // Warm-up run
    status = inner_loop(rt, num_threads, n, n, init_time, vector_add_time);
    if ( !status ) {
        std::cerr << "Warm-up run failed." << std::endl;
        return 1;
    }

    init_time = std::chrono::duration<double>();
    vector_add_time = std::chrono::duration<double>();
    for ( unsigned int iteration = 0; iteration < max_iterations; ++iteration ) {
        status = inner_loop(rt, num_threads, n, n / num_chunks, init_time, vector_add_time);
        if ( !status ) {
            std::cerr << "Run with " << num_chunks << " chunks failed." << std::endl;
            return 1;
        }
    }
    std::cout << "Performance with " << num_threads << " threads, " << num_chunks << " chunks, and n = " << n << std::endl;

    std::cout << "Total time (init): " << init_time.count() << " seconds" << std::endl;
    std::cout << "Average iteration time (init): " << init_time.count() / max_iterations << " seconds" << std::endl;

    std::cout << "Total time: " << vector_add_time.count() << " seconds" << std::endl;
    std::cout << "Average iteration time: " << vector_add_time.count() / max_iterations << " seconds" << std::endl;
    std::cout << "Throughput: " << (ops / vector_add_time.count()) / 1'000'000'000.0 << " GFLOP/s" << std::endl;
    std::cout << "Memory bandwidth: " << (mem / vector_add_time.count()) / 1'000'000'000.0 << " GB/s" << std::endl;
    std::cout << std::endl;

    return 0;
}
