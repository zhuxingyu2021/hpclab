cmake_minimum_required(VERSION 3.16)
project(hpclab1)

set(CMAKE_CXX_STANDARD 14)

add_compile_options("-march=core-avx2")

add_library(MYGEMM SHARED "lib/GEMM_fast.cpp" "lib/GEMM_naive.cpp" "lib/GEMM_utils.cpp")

include_directories("include" "~/devtools/openblas/include")
add_executable (GEMMwop "GEMMwop.cpp")

target_link_libraries(GEMMwop "/home/zhuxingyu/devtools/openblas/lib/libopenblas.a" MYGEMM)

