cmake_minimum_required(VERSION 3.22)

project(computer_vision LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION  ON)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -diag-suppress 20012" )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(TBB REQUIRED) 
find_package(Eigen3 REQUIRED)
find_package (glog 0.6.0 REQUIRED)

add_compile_definitions(TBB_USE_ASSERT)

add_subdirectory(impl/00_naive_mean_curvature 00_naive_mean_curvature)