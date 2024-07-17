cmake_minimum_required(VERSION 3.28)

project(computer_vision LANGUAGES CUDA CXX)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(cmake/Warnings.cmake)
include(cmake/LinkOpen3D.cmake)
include(cmake/AddCommonLibraries.cmake)


# Find clang-tidy
find_program(CLANG_TIDY "clang-tidy")

# If clang-tidy is found, set it up as a compiler option
if(CLANG_TIDY)
    message(STATUS "Found clang-tidy: ${CLANG_TIDY}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY}")
else()
    message(WARNING "clang-tidy not found!")
endif()


add_subdirectory(impl/common common)
add_subdirectory(impl/mean_curvature_01 mean_curvature_01)
add_subdirectory(impl/mean_curvature_02 mean_curvature_02)