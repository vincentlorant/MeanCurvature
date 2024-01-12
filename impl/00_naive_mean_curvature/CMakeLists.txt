cuda_add_executable(00_naive_mean_curvature 
    src/mean_curvature_types.h
    src/cuda_helpers.h
    src/cuda_ring.h
    src/cuda_ring.cu
    src/00_naive_mean_curvature.cu)

target_include_directories(00_naive_mean_curvature PRIVATE
    "../utils")

target_link_libraries(00_naive_mean_curvature 
	glog::glog
	Eigen3::Eigen
	TBB::tbb TBB::tbbmalloc)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(00_naive_mean_curvature PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()



# The options need to be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
if(STATIC_WINDOWS_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Find installed Open3D, which exports Open3D::Open3D
find_package(Open3D REQUIRED)

target_link_libraries(00_naive_mean_curvature Open3D::Open3D)

# On Windows if BUILD_SHARED_LIBS is enabled, copy .dll files to the executable directory
if(WIN32)
    get_target_property(open3d_type Open3D::Open3D TYPE)
    if(open3d_type STREQUAL "SHARED_LIBRARY")
        message(STATUS "Copying Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        add_custom_command(TARGET 00_naive_mean_curvature POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy
                                   ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
                                   ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
    endif()
endif()

