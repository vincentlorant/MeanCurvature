file(GLOB_RECURSE
  common_cc
  CONFIGURE_DEPENDS
  *.cc
)

add_library(common_lib STATIC ${common_cc})

target_include_directories(common_lib PUBLIC
    "."
)

target_add_common_libs(common_lib)