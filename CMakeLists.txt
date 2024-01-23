cmake_minimum_required(VERSION 3.26)

project(computer_vision LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)  
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_SEPARABLE_COMPILATION  ON)


include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LinkOpen3D.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddCommonLibraries.cmake)

add_subdirectory(impl/00_naive_mean_curvature 00_naive_mean_curvature)
add_subdirectory(impl/01_faster_mean_curvature 01_faster_mean_curvature)