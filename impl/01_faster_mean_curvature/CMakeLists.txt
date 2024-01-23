find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(01_faster_mean_curvature_sources
    src/01_faster_mean_curvature.cu)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(01_faster_mean_curvature_headers
        src/mean_curvature_types.h
        src/cuda_helpers.h
        src/cuda_ring.h
        src/host_ring.h)
    
    set(01_faster_mean_curvature_sources
        src/cuda_ring.cu
        src/host_ring.cu
        src/01_faster_mean_curvature.cu)

cuda_add_executable(01_faster_mean_curvature 
    ${01_faster_mean_curvature_headers}
    ${01_faster_mean_curvature_sources})

set_target_properties(01_faster_mean_curvature PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(01_faster_mean_curvature PROPERTIES CUDA_STANDARD 17)

target_include_directories(01_faster_mean_curvature PRIVATE
    "../utils")

target_add_open3d(01_faster_mean_curvature)
target_add_common_libs(01_faster_mean_curvature)
