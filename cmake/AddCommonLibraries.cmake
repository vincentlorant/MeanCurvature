find_package(TBB REQUIRED) 
find_package(Eigen3 REQUIRED)
find_package (glog 0.6.0 REQUIRED)

add_compile_definitions(TBB_USE_ASSERT)

function(target_add_common_libs target_name)
    target_link_libraries(${target_name} 
        glog::glog
        Eigen3::Eigen
        TBB::tbb TBB::tbbmalloc)
endfunction()
