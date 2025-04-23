.. highlight:: c++
   :linenothreshold: 1

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents

   KMM <self>
   install
   api
   Github repository <https://github.com/NLeSC-COMPAS/kmm>

KMM: Kernel Memory Manager
===============

The **Kernel Memory Manager** (KMM) is a lightweight, high-performance framework designed for parallel dataflow execution and efficient memory management on multi-GPU platforms.


KMM automatically manages GPU memory, partitions workloads across multiple GPUs, and schedules tasks efficiently.
Unlike frameworks that require a specific programming model, KMM integrates existing GPU kernels or functions without the need to fully rewrite your code.

Highlights of KMM:

* **Efficient Memory Management**: automatically allocates memory and transfers data between GPU and host only when neccessary.
* **Scalable Computing**: seamlessly spills data from GPU to host memory, enabling huge datasets that exceed GPU memory.
* **Optimized Scheduling**: DAG scheduler automatically tracks dependencies and executes kernels in a sequentially consistent order.
* **Flexible Work Partitioning**: split workloads and data according to user-defined distributions, ensuring utilization of available resources.
* **Portable Execution**: supports existing CUDA, HIP, and CPU-based functions; seamless integration with minimal changes.
* **Multi-Dimensional Arrays**: handles ND-arrays of any shape, dimensionality, and data type.

Basic Example
=============

This example shows how to run a CUDA kernel implementing a vector add operation with KMM.

.. code-block:: cuda

   #include "kmm/kmm.hpp"

   __global__ void vector_add(
       kmm::Range<int64_t> range,
       kmm::GPUSubviewMut<float> output,
       kmm::GPUSubview<float> left,
       kmm::GPUSubview<float> right
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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

