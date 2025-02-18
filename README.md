# KMM: Kernel Memory Manager

[![CPU Build Status](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-multi-compiler.yml/badge.svg)](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-multi-compiler.yml)
[![CUDA Build Status](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-cuda-multi-compiler.yml/badge.svg)](https://github.com/NLeSC-COMPAS/kmm/actions/workflows/cmake-cuda-multi-compiler.yml)

The _Kernel Memory Manager_ (KMM) is a lightweight, high-performance framework designed for parallel dataflow execution and efficient memory management on multi-GPU platforms.
It automatically manages GPU memory, partitions workloads across multiple GPUs, and schedules tasks efficiently.
Unlike frameworks that require a specific programming model, KMM seamlessly integrates existing GPU kernels or functions without the need to rewrite your code.
KMM enables massive workloads in scientific computing, deep learning, and high-performance data processing to run with minimal overhead.


## Features

* *Efficient Memory Management*: Automatically allocates memory and transfers data between GPU and host only when neccessary.
* *Scalable Computing*: Seamlessly “spills” data from the GPU to host memory, enabling execution of large datasets that exceed GPU memory limits.
* *Optimized Task Scheduling*: Uses a DAG scheduler to track dependencies and execute compute kernels in a sequentially consistent order while maximizing parallelism.
* *Flexible Work Partitioning*: Flexible Work Partitioning: Splits workloads and data according to user-defined strategies, ensuring efficient utilization of available resources.
* *Portable Execution*: Portable Execution: Supports CUDA, HIP, and CPU-based functions, allowing seamless integration of existing kernels with minimal code changes.
* *Multi-Dimensional Arrays*: Handles ND-arrays of any shape, dimensionality, and data type.


## Resources

* [Full documentation](https://nlesc-compas.github.io/kmm)


## Example

Example: A simple vector add kernel:

```C++
#include "kmm/kmm.hpp"

__global__ void vector_add(
    kmm::Range<int64_t> range,
    kmm::gpu_subview_mut<float> output,
    kmm::gpu_subview<float> left,
    kmm::gpu_subview<float> right
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x + range.begin;
    if (i >= range.end) return;

    output[i] = left[i] + right[i];
}

int main() {
    // 2B items, 10 chunks, 256 threads per block
    long n = 2'000'000'000;
    long chunk_size = n / 10;
    dim3 block_size = 256;

    // Initialize runtime
    auto rt = kmm::make_runtime();

    // Create arrays
    auto A = kmm::Array<float> {n};
    auto B = kmm::Array<float> {n};
    auto C = kmm::Array<float> {n};

    // Initialize input arrays
    initialize_inputs(A, B);

    // Launch the kernel!
    rt.parallel_submit(
        n, chunk_size,
        kmm::GPUKernel(vector_add, block_size),
        _x,
        write(C[_x]),
        A[_x],
        B[_x]
    );

    // Wait for completion
    rt.synchronize();

    return 0;
}
```


## License

KMM is made available under the terms of the Apache License version 2.0, see the file LICENSE for details.
