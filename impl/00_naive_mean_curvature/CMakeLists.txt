find_package(CUDA REQUIRED)

set(00_naive_mean_curvature_headers
    src/cuda_ring.h)

set(00_naive_mean_curvature_sources
    src/cuda_ring.cu
    ../common/host_ring.cu
    src/00_naive_mean_curvature.cu)

cuda_add_executable(00_naive_mean_curvature 
    ${00_naive_mean_curvature_headers}
    ${00_naive_mean_curvature_sources})

target_include_directories(00_naive_mean_curvature PRIVATE
    "../common")

target_add_open3d(00_naive_mean_curvature)
target_add_common_libs(00_naive_mean_curvature)
