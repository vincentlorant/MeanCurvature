cmake_minimum_required(VERSION 3.28)

project(computer_vision LANGUAGES CUDA CXX)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(cmake/Warnings.cmake)
include(cmake/LinkOpen3D.cmake)
include(cmake/AddCommonLibraries.cmake)

add_subdirectory(impl/common common)
# add_subdirectory(impl/mean_curvature_01 mean_curvature_01)
add_subdirectory(impl/mean_curvature_02 mean_curvature_02)