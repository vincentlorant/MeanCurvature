cmake_minimum_required(VERSION 3.28)

project(computer_vision LANGUAGES CUDA CXX)

# set(CUDACXX /usr/local/cuda/bin/nvcc)
# set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_STANDARD 20)  
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CUDA_SEPARABLE_COMPILATION  ON)

include(cmake/Warnings.cmake)
include(cmake/LinkOpen3D.cmake)
include(cmake/AddCommonLibraries.cmake)

add_subdirectory(impl/common common)
# add_subdirectory(impl/00_naive_mean_curvature 00_naive_mean_curvature)
# add_subdirectory(impl/01_faster_mean_curvature 01_faster_mean_curvature)
add_subdirectory(impl/test test)

# add_subdirectory(___tt test)