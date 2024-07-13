# cuda lib

file(GLOB_RECURSE
  lib_cu
  CONFIGURE_DEPENDS
  cuda/*.cu
)

message(${lib_cu})


add_library(cuda_lib STATIC ${lib_cu})


target_include_directories(common_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/includes/mean_curvature_01
)

target_link_libraries(cuda_lib 
  common_lib
  -lcublas 
)

target_add_common_libs(cuda_lib)

# app

file(GLOB_RECURSE
  app_cc
  CONFIGURE_DEPENDS
  *.cc
)

add_executable(mean_curvature_01 ${app_cc} )

target_link_libraries(mean_curvature_01 common_lib cuda_lib) # 

target_add_open3d(mean_curvature_01)
target_add_common_libs(mean_curvature_01)