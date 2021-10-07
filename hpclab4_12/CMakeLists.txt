cmake_minimum_required(VERSION 3.16)
project(hpclab4_12)

set(CMAKE_CXX_STANDARD 14)

add_compile_options("-march=core-avx2")

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
endif()

include_directories("include" "~/devtools/openblas/include")
add_executable (GEMMopenmp "GEMMopenmp.cpp" "lib/walltime.cpp" "lib/GEMM_fast.cpp" "lib/GEMM_naive.cpp" "lib/GEMM_utils.cpp")
target_link_libraries(GEMMopenmp "/home/zhuxingyu/devtools/openblas/lib/libopenblas.a")


