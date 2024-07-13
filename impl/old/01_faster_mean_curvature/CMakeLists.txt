find_package(CUDA REQUIRED)

set(01_faster_mean_curvature_headers
    ../common/host_ring.h
    src/mean_curvature.h
    src/cuda_ring.h)

set(01_faster_mean_curvature_sources
    ../common/host_ring.cu
    src/cuda_ring.cu
    src/mean_curvature.cu
    src/01_faster_mean_curvature.cc)

add_executable(01_faster_mean_curvature 
    ${01_faster_mean_curvature_headers}
    ${01_faster_mean_curvature_sources})

target_include_directories(01_faster_mean_curvature PRIVATE
    "src"    
    "../common")






# file(GLOB_RECURSE
#     lib_cu
#     CONFIGURE_DEPENDS
#     *.cu
# )

# add_library(01_cuda_lib STATIC ${lib_cu})

# target_include_directories(01_cuda_lib PUBLIC
#     "../common/cuda"
# )

# target_link_libraries(01_cuda_lib 
#   -lcublas 
#   Eigen3::Eigen
#   glog::glog
# )


# add_executable(01_faster_mean_curvature "01_faster_mean_curvature.cc" )


# target_link_libraries(01_faster_mean_curvature 01_cuda_lib)

# target_include_directories(01_faster_mean_curvature PRIVATE
#     "../common"
#     "."
# )



target_add_open3d(01_faster_mean_curvature)
target_add_common_libs(01_faster_mean_curvature)

