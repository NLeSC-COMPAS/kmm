#include "spdlog/spdlog.h"

#include "kmm/kmm.hpp"

void fill_array(kmm::Bounds<2> region, kmm::SubviewMut<float, 2> array, float value) {
    for (auto i = region.x.begin; i < region.x.end; i++) {
        for (auto j = region.y.begin; j < region.y.end; j++) {
            array[i][j] = value;
        }
    }
}

void matrix_multiply(
    kmm::DeviceResource& device,
    kmm::Bounds<3> region,
    int n,
    int m,
    int k,
    kmm::GPUSubviewMut<float, 2> C,
    kmm::GPUSubview<float, 2> A,
    kmm::GPUSubview<float, 2> B
) {
    using kmm::checked_cast;

    float alpha = 1.0;
    float beta = 0.0;

    const float* A_ptr = A.data_at({region.y.begin, region.x.begin});
    const float* B_ptr = B.data_at({region.x.begin, region.z.begin});
    float* C_ptr = C.data_at({region.y.begin, region.z.begin});

#if __CUDA_ARCH__
    KMM_GPU_CHECK(cublasGemmEx(
        device.blas(),
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        checked_cast<int>(region.y.size()),
        checked_cast<int>(region.z.size()),
        checked_cast<int>(region.x.size()),
        &alpha,
        A_ptr,
        CUDA_R_32F,
        checked_cast<int>(A.stride()),
        B_ptr,
        CUDA_R_32F,
        checked_cast<int>(B.stride()),
        &beta,
        C_ptr,
        CUDA_R_32F,
        checked_cast<int>(C.stride()),
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    ));
#elif __HIP_DEVICE_COMPILE__
    KMM_GPU_CHECK(rocblas_gemm_ex(
        device.blas(),
        rocblas_operation_transpose,
        rocblas_operation_transpose,
        checked_cast<int>(region.y.size()),
        checked_cast<int>(region.z.size()),
        checked_cast<int>(region.x.size()),
        &alpha,
        A_ptr,
        rocblas_datatype_f32_r,
        checked_cast<int>(A.stride()),
        B_ptr,
        rocblas_datatype_f32_r,
        checked_cast<int>(B.stride()),
        &beta,
        C_ptr,
        rocblas_datatype_f32_r,
        checked_cast<int>(C.stride()),
        C_ptr,
        rocblas_datatype_f32_r,
        checked_cast<int>(C.stride()),
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0,
        0
    ));
#endif
}

int main() {
    using namespace kmm::placeholders;
    spdlog::set_level(spdlog::level::trace);

    auto rt = kmm::make_runtime();
    int n = 50000;
    int m = 50000;
    int k = 50000;
    int chunk_size = n / 5;

    auto A = kmm::Array<float, 2> {{n, k}};
    auto B = kmm::Array<float, 2> {{k, m}};
    auto C = kmm::Array<float, 2> {{n, m}};

    rt.parallel_submit(
        {n, k},
        {chunk_size, chunk_size},
        kmm::Host(fill_array),
        bounds(_x, _y),
        write(A[_x][_y]),
        1.0F
    );

    rt.parallel_submit(
        {k, m},
        {chunk_size, chunk_size},
        kmm::Host(fill_array),
        bounds(_x, _y),
        write(B[_x][_y]),
        1.0F
    );

    for (size_t repeat = 0; repeat < 1; repeat++) {
        C.reset();

        rt.parallel_submit(
            {k, n, m},
            {chunk_size, chunk_size, chunk_size},
            kmm::GPU(matrix_multiply),
            bounds(_x, _y, _z),
            n,
            m,
            k,
            reduce(kmm::Reduction::Sum, C[_y][_z]),
            A[_y][_x],
            B[_x][_z]
        );

        rt.synchronize();
    }

    return EXIT_SUCCESS;
}
