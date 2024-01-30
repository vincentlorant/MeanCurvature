find_package(CUDA REQUIRED)

set(01_faster_mean_curvature_headers
    ../common/host_ring.h
    src/cuda_ring.h)

set(01_faster_mean_curvature_sources
    src/cuda_ring.cu
    ../common/host_ring.cu
    src/01_faster_mean_curvature.cu)

cuda_add_executable(01_faster_mean_curvature 
    ${01_faster_mean_curvature_headers}
    ${01_faster_mean_curvature_sources})

set_target_properties(01_faster_mean_curvature PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(01_faster_mean_curvature PROPERTIES CUDA_STANDARD 17)

target_include_directories(01_faster_mean_curvature PRIVATE
    "src"    
    "../common")

target_add_open3d(01_faster_mean_curvature)
target_add_common_libs(01_faster_mean_curvature)
