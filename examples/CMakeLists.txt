add_executable(vector_add_example ${PROJECT_SOURCE_DIR}/examples/vector_add.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            vector_add_example
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/vector_add.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(vector_add_example PRIVATE cxx_std_17)
target_link_libraries(vector_add_example PRIVATE kmm)

add_executable(cpp_threads_vector_add_example ${PROJECT_SOURCE_DIR}/examples/cpp_threads_vector_add.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            cpp_threads_vector_add_example
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/cpp_threads_vector_add.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(cpp_threads_vector_add_example PRIVATE cxx_std_17)
target_link_libraries(cpp_threads_vector_add_example PRIVATE kmm)

add_executable(point_in_poly_example ${PROJECT_SOURCE_DIR}/examples/point_in_poly.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            point_in_poly_example
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/point_in_poly.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(point_in_poly_example PRIVATE cxx_std_17)
target_link_libraries(point_in_poly_example PRIVATE kmm)


add_executable(matrix_multiply_example ${PROJECT_SOURCE_DIR}/examples/matrix_multiply.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            matrix_multiply_example
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/matrix_multiply.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(matrix_multiply_example PRIVATE cxx_std_17)
target_link_libraries(matrix_multiply_example PRIVATE kmm)


add_executable(histogram_example ${PROJECT_SOURCE_DIR}/examples/histogram.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            histogram_example
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/histogram.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(histogram_example PRIVATE cxx_std_17)
target_link_libraries(histogram_example PRIVATE kmm)


add_executable(reduction_example ${PROJECT_SOURCE_DIR}/examples/reduction.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            reduction_example
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/reduction.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(reduction_example PRIVATE cxx_std_17)
target_link_libraries(reduction_example PRIVATE kmm)


add_executable(structs_example ${PROJECT_SOURCE_DIR}/examples/structs.cu)
if(KMM_USE_CUDA)
    set_target_properties(
            structs_example
            PROPERTIES
            CUDA_ARCHITECTURES "80"
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
if(KMM_USE_HIP)
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/structs.cu PROPERTIES LANGUAGE HIP)
endif()
target_compile_features(structs_example PRIVATE cxx_std_17)
target_link_libraries(structs_example PRIVATE kmm)