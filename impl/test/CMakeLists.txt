# cuda lib


file(GLOB_RECURSE
  lib_cu
  CONFIGURE_DEPENDS
  cuda/*.cu
)

message(${lib_cu})


add_library(cuda_lib STATIC ${lib_cu})


target_include_directories(cuda_lib PUBLIC
    "../common"
    "cuda"
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

add_executable(test ${app_cc} )

target_include_directories(test PRIVATE
    "../common"
    "."
    )

target_link_libraries(test cuda_lib)

target_add_open3d(test)
target_add_common_libs(test)