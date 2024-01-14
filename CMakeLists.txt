cmake_minimum_required(VERSION 3.22)

project(computer_vision LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_SEPARABLE_COMPILATION  ON)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})


include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/LinkOpen3D.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddCommonLibraries.cmake)

add_subdirectory(impl/00_naive_mean_curvature 00_naive_mean_curvature)