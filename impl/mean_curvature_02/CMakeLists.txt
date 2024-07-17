# cuda lib

file(GLOB_RECURSE
  lib_cu
  CONFIGURE_DEPENDS
  cuda/*.cu
)

add_library(cuda_lib_02 STATIC ${lib_cu})

target_include_directories(cuda_lib_02 PUBLIC
    ${CMAKE_SOURCE_DIR}/includes/mean_curvature_02
)

target_link_libraries(cuda_lib_02 
  common_lib
  -lcublas 
)

target_add_common_libs(cuda_lib_02)

# app

file(GLOB_RECURSE
  app_cc
  CONFIGURE_DEPENDS
  *.cc
)

add_executable(mean_curvature_02 ${app_cc} )

target_link_libraries(mean_curvature_02 common_lib cuda_lib_02) # 

target_add_open3d(mean_curvature_02)
target_add_common_libs(mean_curvature_02)