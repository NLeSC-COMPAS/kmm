add_executable(vector_add_bench ${PROJECT_SOURCE_DIR}/benchmarks/vector_add.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            vector_add_bench
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/benchmarks/vector_add.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(vector_add_bench PRIVATE cxx_std_17)
target_link_libraries(vector_add_bench PRIVATE kmm)
